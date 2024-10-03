"""
title: Cost Tracker
description: This function is designed to manage and calculate the costs associated with user interactions and model usage in a Open WebUI appliance.
author: bgeneto
author_url: https://github.com/bgeneto/open-webui-cost-tracker
funding_url: https://github.com/open-webui
version: 0.1.3
license: MIT
requirements: requests, tiktoken, cachetools, pydantic
environment_variables:
"""

import hashlib
import json
import os
import time
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from threading import Lock
from typing import Any, Awaitable, Callable, Optional

import requests
import tiktoken
from cachetools import TTLCache, cached
from open_webui.utils.misc import get_last_assistant_message, get_messages_content
from pydantic import BaseModel, Field


class Config:
    DATA_DIR = "data"
    CACHE_DIR = os.path.join(DATA_DIR, ".cache")
    USER_COST_FILE = os.path.join(DATA_DIR, f"costs-{datetime.now().year}.json")
    CACHE_TTL = 432000  # try to keep model pricing json file for 5 days in the cache.
    CACHE_MAXSIZE = 2
    DECIMALS = "0.000001"
    DEBUG = False


# Initialize cache
cache = TTLCache(maxsize=Config.CACHE_MAXSIZE, ttl=Config.CACHE_TTL)


def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        if Config.DEBUG:
            print(f"**WARN: Encoding for model {model} not found. Using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")


class UserCostManager:
    def __init__(self, cost_file_path):
        self.cost_file_path = cost_file_path
        self._ensure_cost_file_exists()

    def _ensure_cost_file_exists(self):
        if not os.path.exists(self.cost_file_path):
            with open(self.cost_file_path, "w", encoding="UTF-8") as cost_file:
                json.dump({}, cost_file)

    def _read_costs(self):
        with open(self.cost_file_path, "r", encoding="UTF-8") as cost_file:
            return json.load(cost_file)

    def _write_costs(self, costs):
        with open(self.cost_file_path, "w", encoding="UTF-8") as cost_file:
            json.dump(costs, cost_file, indent=4)

    def update_user_cost(
        self, user_email, model, input_tokens, output_tokens, total_cost
    ):
        costs = self._read_costs()
        timestamp = datetime.now().isoformat()

        if user_email not in costs:
            costs[user_email] = []

        costs[user_email].append(
            {
                "model": model,
                "timestamp": timestamp,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": round(total_cost, 6),
            }
        )

        self._write_costs(costs)


class ModelCostManager:
    def __init__(self, cache_dir=Config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.lock = Lock()
        self.url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
        self.cache_file_path = self._get_cache_filename()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_filename(self):
        cache_file_name = hashlib.sha256(self.url.encode()).hexdigest() + ".json"
        return os.path.normpath(os.path.join(self.cache_dir, cache_file_name))

    def _is_cache_valid(self, cache_file_path):
        cache_file_mtime = os.path.getmtime(cache_file_path)
        return time.time() - cache_file_mtime < cache.ttl

    @cached(cache=cache)
    def get_cost_data(self):
        """
        Fetches a JSON file from a URL and stores it in cache.

        This method attempts to retrieve a JSON file from the specified URL. To optimize performance and reduce
        network requests, it caches the JSON data locally. If the cached data is available and still valid,
        it returns the cached data instead of making a new network request. If the cached data is not available
        or has expired, it fetches the data from the URL, caches it, and then returns it.

        Returns:
            dict: The JSON data retrieved from the URL or the cache.

        Raises:
            requests.RequestException: If the network request fails and no valid cache is available.
        """

        with self.lock:
            if os.path.exists(self.cache_file_path) and self._is_cache_valid(
                self.cache_file_path
            ):
                with open(self.cache_file_path, "r", encoding="utf-8") as cache_file:
                    return json.load(cache_file)
        try:
            print("**DEBUG: Downloading model cost file!")
            response = requests.get(self.url)
            response.raise_for_status()
            data = response.json()

            with self.lock:
                with open(self.cache_file_path, "w", encoding="utf-8") as cache_file:
                    json.dump(data, cache_file)

            return data
        except requests.RequestException as e:
            print(
                f"**ERROR: Failed to download file from {self.url}. Using cached file if available. Error: {e}"
            )
            with self.lock:
                if os.path.exists(self.cache_file_path):
                    with open(
                        self.cache_file_path, "r", encoding="utf-8"
                    ) as cache_file:
                        return json.load(cache_file)
                else:
                    raise e

    def get_model_data(self, model):
        return self.get_cost_data().get(model, {})


class CostCalculator:
    def __init__(
        self, user_cost_manager: UserCostManager, model_cost_manager: ModelCostManager
    ):
        self.model_cost_manager = model_cost_manager
        self.user_cost_manager = user_cost_manager

    def calculate_costs(self, model, input_tokens, output_tokens, compensation):
        model_pricing_data = self.model_cost_manager.get_model_data(model)
        input_cost_per_token = Decimal(
            str(model_pricing_data.get("input_cost_per_token", 0))
        )
        output_cost_per_token = Decimal(
            str(model_pricing_data.get("output_cost_per_token", 0))
        )

        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        total_cost = Decimal(float(compensation)) * (input_cost + output_cost)
        total_cost = total_cost.quantize(
            Decimal(Config.DECIMALS), rounding=ROUND_HALF_UP
        )

        return total_cost


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=15, description="Priority level")
        compensation: float = Field(
            default=1.0, description="Compensation for price calculation (percent)"
        )
        debug: bool = Field(default=False, description="Display debugging messages")
        elapsed_time: bool = Field(default=True, description="Display the elapsed time")
        number_of_tokens: bool = Field(
            default=True, description="Display total number of tokens"
        )
        tokens_per_sec: bool = Field(
            default=True, description="Display tokens per second metric"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.model_cost_manager = ModelCostManager()
        self.user_cost_manager = UserCostManager(Config.USER_COST_FILE)
        self.cost_calculator = CostCalculator(
            self.user_cost_manager, self.model_cost_manager
        )
        self.start_time = None
        self.input_tokens = 0

    def _sanitize_model_name(self, model):
        name = model
        if name.startswith("anthropic."):
            name = name.replace("anthropic.", "")
        if name.startswith("google_genai."):
            name = name.replace("google_genai.", "")
        if name.endswith("-tuned"):
            name = name[:-6]
        if name.endswith("-latest"):
            name = name[:-7]
        return name.lower().strip()

    def _remove_roles(self, content):
        # Define the roles to be removed
        roles = ["SYSTEM:", "USER:", "ASSISTANT:", "PROMPT:"]

        # Process each line
        def process_line(line):
            for role in roles:
                if line.startswith(role):
                    return line.split(":", 1)[1].strip()
            return line  # Return the line unchanged if no role matches

        return "\n".join([process_line(line) for line in content.split("\n")])

    def _get_model_and_encoding(self, body):
        model = "gpt-4o"
        if "model" in body:
            model = self._sanitize_model_name(body["model"])
        enc = get_encoding(model)
        return model, enc

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        model, enc = self._get_model_and_encoding(body)
        input_content = self._remove_roles(
            get_messages_content(body["messages"])
        ).strip()
        self.input_tokens = len(enc.encode(input_content))

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Processing {self.input_tokens} input tokens...",
                    "done": False,
                },
            }
        )

        self.start_time = time.time()

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        model, enc = self._get_model_and_encoding(body)
        output_tokens = len(enc.encode(get_last_assistant_message(body["messages"])))

        total_cost = self.cost_calculator.calculate_costs(
            model, self.input_tokens, output_tokens, self.valves.compensation
        )

        if __user__:
            if "email" in __user__:
                user_email = __user__["email"]
            else:
                print("**WARN: User email not found!")
            try:
                self.user_cost_manager.update_user_cost(
                    user_email,
                    model,
                    self.input_tokens,
                    output_tokens,
                    float(total_cost),
                )
            except Exception as _:
                print("**ERROR: Unable to update user cost file!")
        else:
            print("**WARN: User not found!")

        tokens = self.input_tokens + output_tokens
        tokens_per_sec = tokens / elapsed_time
        stats_array = []

        if self.valves.elapsed_time:
            stats_array.append(f"{elapsed_time:.2f} sec")
        if self.valves.tokens_per_sec:
            stats_array.append(f"{tokens_per_sec:.2f} T/s")
        if self.valves.number_of_tokens:
            stats_array.append(f"{tokens} tokens")

        if float(total_cost) < float(Config.DECIMALS):
            stats_array.append(f"${total_cost:.2f}")
        else:
            stats_array.append(f"${total_cost:.6f}")

        stats = " | ".join(stats_array)

        await __event_emitter__(
            {"type": "status", "data": {"description": stats, "done": True}}
        )

        return body