"""
title: User Cost Tracker
description: This valve tracks user token usage and cost of all LLM models and outputs the cost and tokens to a json file.
author: bgeneto
version: 0.3.2
license: MIT
requirements: pydantic, requests, tiktoken
environment_variables:
"""

import json
import math
import os
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import List, Optional

import requests
import tiktoken
from pydantic import BaseModel, Field
from schemas import OpenAIChatMessage


class ModelPricing:
    def __init__(self, json_url, json_file_path):
        self.json_url = json_url
        self.json_file_path = json_file_path
        self.pricing_data = self._load_pricing_data()

    def _load_pricing_data(self):
        if not os.path.exists(self.json_file_path):
            self._download_pricing_data()

        with open(self.json_file_path, "r", encoding="UTF-8") as json_file:
            return json.load(json_file)

    def _download_pricing_data(self):
        response = requests.get(self.json_url, timeout=10)
        response.raise_for_status()
        with open(self.json_file_path, "wb") as json_file:
            json_file.write(response.content)

    def get_model_data(self, model):
        return self.pricing_data.get(model, {})


class UserCostTracker:

    def __init__(self, cost_file_path):
        self.cost_file_path = cost_file_path
        self._ensure_cost_file_exists()
        self.decimals = "0.000001"

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

    def update_user_cost(self, user_email, model, input_tokens, output_tokens, cost):
        costs = self._read_costs()
        timestamp = datetime.now().isoformat()

        if user_email not in costs:
            costs[user_email] = []

        # Round the cost to 6 decimal places
        cost = round(cost, 6)

        # If the cost is less than threshold, consider it as zero and don't write it to the file
        if cost >= float(self.decimals):
            costs[user_email].append(
                {
                    "model": model,
                    "timestamp": timestamp,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                }
            )

        self._write_costs(costs)


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        cost_file: str = ""

    def __init__(self):
        self.type = "filter"
        self.name = "User Cost Tracker"
        output_dir = os.path.dirname(__file__)
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "priority": 3,
                "cost_file": os.path.normpath(
                    os.path.join(output_dir, "user_costs.json")
                ),
            }
        )
        json_file_path = os.path.join(
            output_dir, "model_prices_and_context_windows.json"
        )
        json_url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

        self.model_pricing = ModelPricing(json_url, json_file_path)
        self.user_cost_tracker = UserCostTracker(self.valves.cost_file)

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        pass

    def _deduplicate_files(self, data):
        if not isinstance(data, dict) or "files" not in data:
            return data

        files = data["files"]
        seen_collection_names = set()
        deduplicated_files = [
            file
            for file in files
            if file["type"] == "file"
            and file["collection_name"] not in seen_collection_names
            and not seen_collection_names.add(file["collection_name"])
        ]

        data["files"] = deduplicated_files
        return data

    def _calculate_tokens(self, messages, enc):
        return sum(len(enc.encode(message["content"])) for message in messages)

    def _sanitize_model_name(self, model):
        name = model
        if name.startswith("anthropic."):
            # remove the prefix to match the model names in the pricing data
            name = name.replace("anthropic.", "")
        if name.startswith("google_genai."):
            # remove the prefix
            name = name.replace("google_genai.", "")
        # check if the model exists in the pricing data
        if name not in self.model_pricing.pricing_data:
            if name.endswith("-tuned"):
                # remove the -tuned suffix
                name = name[:-6]
            if name.endswith("-latest"):
                # remove the -latest suffix
                name = name[:-7]
        return name.lower().strip()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        body = self._deduplicate_files(body)
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        model = self._sanitize_model_name(body.get("model"))

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                enc = tiktoken.encoding_for_model("gpt-4o")
            except KeyError:
                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        model_data = self.model_pricing.get_model_data(model)
        if model_data:
            input_tokens = 0
            output_tokens = 0

            # Calculate input tokens for all messages except the last one
            for message in messages[:-1]:
                input_tokens += len(enc.encode(message["content"]))

            # Calculate output tokens for the last message if it's an assistant response
            if messages and messages[-1]["role"] == "assistant":
                output_tokens = len(enc.encode(messages[-1]["content"]))

            input_cost_per_token = Decimal(
                str(model_data.get("input_cost_per_token", 0))
            )
            output_cost_per_token = Decimal(
                str(model_data.get("output_cost_per_token", 0))
            )

            input_cost = (
                math.ceil(1.2 * input_tokens) * input_cost_per_token
            )  # 20% extra for input to match OpenAI's pricing
            output_cost = output_tokens * output_cost_per_token
            total_cost = input_cost + output_cost

            # Round the total cost to a given decimal places for monetary values
            total_cost = total_cost.quantize(
                Decimal(self.user_cost_tracker.decimals), rounding=ROUND_HALF_UP
            )

            if user:
                user_email = user.get("email")
                self.user_cost_tracker.update_user_cost(
                    user_email, model, input_tokens, output_tokens, float(total_cost)
                )
        else:
            print(f"Model {model} not found in pricing data.")

        return body
