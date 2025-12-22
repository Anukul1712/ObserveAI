import json
import logging
import re
import asyncio
from pathlib import Path
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import Config

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classifies user queries into a SecondaryIntent using an LLM.
    """

    def __init__(self, llm):
        self.llm = llm
        self.taxonomy = self._load_taxonomy()
        self.intents_list = self._flatten_intents()

    def _load_taxonomy(self) -> dict:
        taxonomy_path = Config.DATA_DIR / "taxonomy.json"
        with open(taxonomy_path, "r") as f:
            return json.load(f)

    def _flatten_intents(self) -> List[str]:
        """Extracts all secondary intents from the taxonomy."""
        intents = []
        for primary, roles in self.taxonomy.items():
            for role, intent_list in roles.items():
                intents.extend(intent_list)
        return list(set(intents))

    def classify(self, query: str) -> str:
        """
        Classifies the query into one of the known intents.
        """
        if not self.intents_list:
            logger.warning("No intents loaded. Returning 'Unknown'.")
            return "Unknown"

        prompt_template = """
        You are an expert intent classifier for a customer support system.
        Your task is to map the user's query to the most relevant 'Secondary Intent' from the provided list.

        List of Valid Intents:
        {intents}
        
        User Query: "{query}"
        
        Instructions:
        1. Analyze the user query carefully.
        2. Select exactly ONE intent from the list that best matches the query.
        3. Output ONLY the intent name. Do not include any explanation.
        
        Intent:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["intents", "query"]
        )

        chain = prompt | self.llm | StrOutputParser()

        # Pass the list as a comma-separated string or a formatted list
        intents_str = ", ".join(self.intents_list)
        result = chain.invoke({"intents": intents_str, "query": query})
        cleaned_result = result.strip()

        logger.info(f"Classified Intent: {cleaned_result}")
        # Validation
        if cleaned_result in self.intents_list:
            return cleaned_result
        else:
            logger.warning(
                f"LLM returned invalid intent: {cleaned_result}")
            return "Unknown"

    async def aclassify(self, query: str) -> str:
        """
        Asynchronously classifies the query into one of the known intents.
        """
        if not self.intents_list:
            logger.warning("No intents loaded. Returning 'Unknown'.")
            return "Unknown"

        prompt_template = """
        You are an expert intent classifier for a customer support system.
        Your task is to map the user's query to the most relevant 'Secondary Intent' from the provided list.

        List of Valid Intents:
        {intents}
        
        User Query: "{query}"
        
        Instructions:
        1. Analyze the user query carefully.
        2. Select exactly ONE intent from the list that best matches the query.
        3. Output ONLY the intent name. Do not include any explanation.
        
        Intent:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["intents", "query"]
        )

        chain = prompt | self.llm | StrOutputParser()

        # Pass the list as a comma-separated string or a formatted list
        intents_str = ", ".join(self.intents_list)
        result = await chain.ainvoke({"intents": intents_str, "query": query})
        cleaned_result = result.strip()

        # Validation
        if cleaned_result in self.intents_list:
            return cleaned_result
        else:
            logger.warning(
                f"LLM returned invalid intent: {cleaned_result}")
            return "Unknown"

    async def aclassify_batch(self, queries: List[str]) -> List[str]:
        """
        Classifies a batch of queries in parallel.
        Waits for all to complete before returning.
        """
        tasks = [self.aclassify(q) for q in queries]
        return await asyncio.gather(*tasks)
