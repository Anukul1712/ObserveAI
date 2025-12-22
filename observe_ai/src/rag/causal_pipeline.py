import json
import numpy as np
import pandas as pd
import joblib
import os
import logging
import re
from langchain_core.output_parsers import JsonOutputParser
from src.config import Config
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class CausalPipeline:
    def __init__(self, llm):
        # Path declarations
        # Assuming this file is in src/rag/, so we go up two levels to observe_ai/
        base_dir = (Config.BASE_DIR)
        # base_dir = Path(__file__).parent.parent.parent

        self.data_path = Path(base_dir) / "causality" / \
            "NewObservationalDataset.csv"
        self.schema_path = Path(base_dir) / "causality" / \
            "variable_schema.json"
        self.scm_bundle_path = Path(
            base_dir) / "causality" / "scm_xgboost_gpu.pkl"

        self.llm = llm

        logger.info("CausalPipeline: Loading resources...")
        # Load resources
        logger.info("Loading schema...")
        self.schema = self._load_schema()
        self.schema_by_name = {v["name"]: v for v in self.schema}
        self.numeric_vars = [v["name"] for v in self.schema]

        logger.info("Loading dataset...")
        self.df_full = self._load_dataset()
        # Ensure all numeric vars exist
        missing = [c for c in self.numeric_vars if c not in self.df_full.columns]
        if missing:
            logger.warning(
                f"WARNING: These schema vars not in dataset: {missing}")

        self.df_numeric = self.df_full[self.numeric_vars].copy()

        logger.info("Loading SCM bundle...")
        self.bundle = self._load_scm_bundle()
        logger.info("CausalPipeline initialized.")

    def _load_schema(self):
        with open(self.schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_dataset(self):
        return pd.read_csv(self.data_path)

    def _load_scm_bundle(self):
        return joblib.load(self.scm_bundle_path)

    def simulate_scm(self, base_row, interventions):
        """
        base_row: pd.Series with numeric vars only.
        interventions: dict { var_name: new_value }
        """
        if self.bundle is None:
            raise ValueError("SCM Bundle is not loaded.")

        scm_models = self.bundle["scm_models"]
        topo_order = self.bundle["topo_order"]

        x = base_row.copy()

        # apply interventions
        for var, val in interventions.items():
            if var not in x.index:
                raise ValueError(
                    f"Intervened variable '{var}' not found in base_row.")
            x[var] = val

        # propagate
        for node in topo_order:
            if node in interventions:
                continue  # intervened variables stay fixed

            model_info = scm_models.get(node, None)
            if model_info is None:
                continue  # root or missing model

            parents = model_info["parents"]
            model = model_info["model"]
            parent_vals = np.array(
                [x[p] for p in parents], dtype=float).reshape(1, -1)
            x[node] = float(model.predict(parent_vals)[0])

        return x

    def make_query_interpret_prompt(self, user_query):
        def build_schema_text(schema):
            lines = []
            for v in schema:
                lines.append(
                    f"- name: {v['name']}\n"
                    f"  display_name: {v['display_name']}\n"
                    f"  role: {v['role']}\n"
                    f"  description: {v['description']}\n"
                )
            return "\n".join(lines)

        schema_text = build_schema_text(self.schema)

        prompt = f"""You are a causal query interpreter.

User's question:
\"\"\"{user_query}\"\"\"

You are given a list of numeric variables you can intervene on or measure:

{schema_text}

Your job:
1. Decide which variables the user implicitly wants to INTERVENE on (change), if any.
2. Decide which variables should be the TARGET outcomes of interest.
3. For each intervention, specify:
- "variable": one of the 'name' fields from the list above (use exact name)
- "direction": "increase", "decrease", or "set"
- "magnitude": "small", "medium", "large", or "exact"

Important reasoning rules:
- You must ALWAYS return at least one intervention and at least one target.
- If the user’s question does NOT clearly indicate which variables are relevant, select the most weakly or slightly related variables.
- If the direction of change (increase/decrease/set) is unclear, determine the most informative direction by reasoning counterfactually — i.e., choose the direction that would help answer the question if we imagined an alternate scenario where that variable changed.

Additional rules:
- Use ONLY variable names from the provided list for "variable" and in "targets".
- Do NOT invent new variable names.
- Never return empty lists for "interventions" or "targets".
- Always output valid JSON.

Output STRICTLY in this JSON format and nothing else:

{{
"interventions": [
    {{
    "variable": "name_here",
    "direction": "increase|decrease|set",
    "magnitude": "small|medium|large|exact"
    }}
],
"targets": ["var_name_1", "var_name_2", "..."]
}}"""
        return prompt

    def call_with_json_parser(self, prompt):
        """
        Call LLM with JSON output parser for structured output.
        """
        # Create a chain with the LLM and parser
        chain = self.llm | JsonOutputParser()

        try:
            result = chain.invoke(prompt)
            return result
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            # Fallback to manual extraction
            response = self.llm.invoke(prompt)
            return self._manual_json_extract(response.content)  # type: ignore

    def _manual_json_extract(self, raw_output: str):
        """
        Extract the single JSON plan object from the model output.
        """
        segment = raw_output

        # 1. Keep only text after the LAST "Answer:"
        if "Answer:" in segment:
            segment = segment.split("Answer:")[-1]

        # 2. Find JSON boundaries
        start = segment.find("{")
        end = segment.rfind("}")

        if start == -1 or end == -1 or end <= start:
            # Fallback: try to find json block
            match = re.search(r'```json\s*(\{.*?\})\s*```', segment, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                raise ValueError(
                    "Could not find a JSON object in model output.")
        else:
            json_str = segment[start:end+1]

        # 3. Clean potential smart quotes (just in case)
        json_str = (json_str.replace("“", '"')
                    .replace("”", '"')
                    .replace("‘", '"')
                    .replace("’", '"'))

        # 4. Parse JSON
        try:
            plan = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common JSON errors if needed, or just fail
            raise ValueError(f"Failed to parse JSON: {json_str}")

        return plan

    def magnitude_to_new_value(self, current_val, var_meta, magnitude, direction):
        """
        Convert (direction, magnitude) into a numeric value for intervention.
        """
        name = var_meta["name"]
        vmin = var_meta.get("scale_min", float(self.df_numeric[name].min()))
        vmax = var_meta.get("scale_max", float(self.df_numeric[name].max()))
        span = vmax - vmin if vmax > vmin else 1.0

        # If direction is "set", we choose an absolute target value.
        if direction == "set":
            if magnitude == "exact":
                # Use dataset median as a neutral "exact" target
                base = float(self.df_numeric[name].median())
            elif magnitude == "small":
                base = vmin + 0.25 * span
            elif magnitude == "medium":
                base = vmin + 0.50 * span
            elif magnitude == "large":
                base = vmin + 0.75 * span
            else:
                base = float(self.df_numeric[name].mean())
            return float(np.clip(base, vmin, vmax))

        # For increase / decrease, we use a step around the current value
        if magnitude == "small":
            step = 0.15 * span
        elif magnitude == "medium":
            step = 0.30 * span
        elif magnitude == "large":
            step = 0.50 * span
            step = min(step, span)  # clamp
        else:
            step = 0.20 * span

        if direction == "decrease":
            step = -step

        new_val = current_val + step
        return float(np.clip(new_val, vmin, vmax))

    def run(self, user_query: str, transcript_ids: List[str]):
        prompt = self.make_query_interpret_prompt(user_query)

        try:
            plan = self.call_with_json_parser(prompt)
        except Exception as e:
            logger.error(f"Error extracting plan: {e}")
            return f"Error interpreting query: {e}"

        results_for_llm = []
        interventions_spec = plan.get("interventions", [])
        target_vars = plan.get("targets", [])

        for tid in transcript_ids:
            # find row for this transcript
            matches = self.df_full[self.df_full["transcript_id"] == tid]
            if matches.empty:
                logger.warning(
                    f"WARNING: transcript_id {tid} not found in dataset.")
                continue

            base_full = matches.iloc[0]
            base_numeric = base_full[self.numeric_vars]

            interventions_numeric = {}

            for inter in interventions_spec:
                var = inter["variable"]
                direction = inter.get("direction", "increase")
                magnitude = inter.get("magnitude", "small")

                if var not in self.numeric_vars:
                    logger.warning(
                        f"WARNING: intervention var {var} not a known numeric variable. Skipping.")
                    continue

                var_meta = self.schema_by_name[var]
                cur_val = float(base_numeric[var])

                new_val = self.magnitude_to_new_value(
                    cur_val, var_meta, magnitude, direction)
                interventions_numeric[var] = new_val

            if not interventions_numeric:
                logger.info(
                    f"No valid interventions for transcript_id {tid}. Skipping SCM.")
                # Even if no interventions, we might want to report current values if it's descriptive
                if plan.get("query_type") == "descriptive":
                    # Just report targets
                    pass
                else:
                    continue

            # simulate SCM
            if interventions_numeric:
                cf_numeric = self.simulate_scm(
                    base_numeric, interventions_numeric)
            else:
                cf_numeric = base_numeric  # No change

            # collect before/after for targets
            target_summaries = []
            for tvar in target_vars:
                if tvar not in self.numeric_vars:
                    logger.warning(
                        f"WARNING: target var {tvar} not numeric. Skipping.")
                    continue
                before = float(base_numeric[tvar])
                after = float(cf_numeric[tvar])
                target_summaries.append({
                    "variable": tvar,
                    "before": before,
                    "after": after,
                    "delta": after - before,
                })

            results_for_llm.append({
                "transcript_id": tid,
                "query_type": plan.get("query_type", "counterfactual"),
                "interventions": interventions_numeric,
                "targets": target_summaries,
            })

        return self.format_results_for_llm(results_for_llm)

    def format_results_for_llm(self, results):
        output_lines = []
        output_lines.append("Causal Analysis:")

        if len(results) == 0:
            output_lines.append("No interventions found.")
            return "\n".join(output_lines)

        for item in results:
            output_lines.append(f"\nCase ID: {item['transcript_id']}")
            output_lines.append(f"Query Type: {item['query_type']}")

            if item['interventions']:
                output_lines.append("Interventions:")
                for k, v in item["interventions"].items():
                    output_lines.append(f"  - {k}: {v:.4f}")

            if item['targets']:
                output_lines.append("Targets (Before -> After, Delta):")
                for t in item["targets"]:
                    output_lines.append(
                        f"  - {t['variable']}: {t['before']:.4f} -> {t['after']:.4f} (Delta: {t['delta']:.4f})")
            else:
                output_lines.append("No targets specified.")

        return "\n".join(output_lines)


if __name__ == "__main__":
    from langchain_deepseek import ChatDeepSeek
    query = "Why are refund approvals often delayed?"
    transcript_ids = ['2c4f9b99-8f1d-44fd-9c36-5ffc8f94a7e4',
                    '121b3588-ef11-4242-9906-94051686d905',
                    '121b3588-ef11-4242-9906-94051686d905',
                    '20f191f5-5302-4dc8-8cce-5c41ba28c0a0',
                    '5c70decd-5b12-4712-9f50-8828751e5b0b',
                    '20f191f5-5302-4dc8-8cce-5c41ba28c0a0',
                    '4e2a9f09-bc1d-43c0-b4af-e8f31469adde',
                    '6634c1b7-2e41-42ca-a45d-a755a9c0c3a6',
                    '490f2928-42e3-47a9-b749-e7710796e376',
                    '0bcae6fa-b069-419b-b59c-93e8b9c5b0ac',
                    '6604bc80-466a-45dd-b463-3f3936103fff',
                    '15eba159-d35b-491d-b374-821dbef46a74',
                    '3320b20a-8edd-45a1-88d9-87fd2ff758a7',
                    '3320b20a-8edd-45a1-88d9-87fd2ff758a7',
                    '80cb6c6b-a855-435b-b2fb-096bbeea12dc',
                    '6abf0197-ebfd-4683-b015-f5db5c6b23a3',
                    '2d6821e8-d1ad-4cc1-9a75-cd1b076d8539',
                    '3fd535f5-4209-4d9d-ad9a-cfb1890813e0',
                    '5c207293-78e3-4ea0-a1c5-ebddde9b9c29',
                    '9d2fd3db-53eb-4eb5-b99a-9c51cffa7114',
                    '2be537a2-5982-4793-a2a5-68294b4e1ff0',
                    '991e0712-5685-41c7-9527-c88418742fb6',
                    '5c70decd-5b12-4712-9f50-8828751e5b0b',
                    '7752e117-c1c4-4fa5-afaa-20c571a6cde9',
                    '6c4dc4c7-0231-4fcd-b344-888fab97b240',
                    '20f191f5-5302-4dc8-8cce-5c41ba28c0a0',
                    '15eba159-d35b-491d-b374-821dbef46a74',
                    'c329c519-f40a-4385-aa98-92ee13b4cbb3',
                    'c329c519-f40a-4385-aa98-92ee13b4cbb3',
                    '83f0333d-2325-4ba6-9ae4-a1317b0a9a02',
                    'aa910782-b69d-4146-9460-afbd50775d48',
                    '43e0f95d-8b25-4d25-9201-48e91cac7d1a',
                    'a24d3b44-aa16-4166-bd3b-9350f6546e59',
                    '44f98161-ba24-4085-b135-5f1002188592']

    llm = ChatDeepSeek(
        model="deepseek/deepseek-v3.2",
        api_key=Config.OPEN_ROUTER_KEY,
        # note: use `api_base`, not `base_url`
        api_base="https://openrouter.ai/api/v1",
        extra_body={"reasoning": {"enabled": False},
                    "usage": {"include": True}},  # optional — e.g. enable reasoning tokens
        temperature=0.0,
        max_tokens=10000,
    )
    causal = CausalPipeline(llm)
    result = causal.run(query, transcript_ids)
    print(result)
