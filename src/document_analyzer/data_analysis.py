"""
This module defines the DocumentAnalyzer class for automated document analysis using a language model (LLM).

Overview:
- Loads a pre-trained LLM via ModelLoader.
- Uses prompts and output parsers to extract structured metadata and summaries from document text.
- Employs robust error handling and logging for transparency.

Key Components:
1. Imports:
   - Standard libraries (os, sys) for system operations.
   - Custom modules for model loading, logging, exception handling, and data models.
   - LangChain output parsers for converting LLM output to structured JSON.
   - PROMPT_REGISTRY for prompt templates.

2. DocumentAnalyzer Class:
   - Initializes model, parsers, and prompt for analysis.
   - All actions are logged.
   - Handles initialization errors via custom exceptions.

3. analyze_document Method:
   - Chains prompt, LLM, and fixing parser for robust metadata extraction.
   - Invokes the chain with formatting instructions and document text.
   - Returns parsed metadata as a dictionary.
   - Errors are logged and raised as DocumentPortalException.

Usage Example:
    analyzer = DocumentAnalyzer()
    metadata = analyzer.analyze_document(document_text)

Extensibility:
- Easily adaptable for different prompts, models, or output structures.

"""

import os
import sys
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY # type: ignore

class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """
    def __init__(self):
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm()
            
            # Prepare parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            self.prompt = PROMPT_REGISTRY["document_analysis"]
            
            log.info("DocumentAnalyzer initialized successfully")
            
            
        except Exception as e:
            log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)
        
        
    def analyze_document(self, document_text:str)-> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
        """
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            
            log.info("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            log.info("Metadata extraction successful", keys=list(response.keys()))
            
            return response

        except Exception as e:
            log.error("Metadata analysis failed", error=str(e))
            raise DocumentPortalException("Metadata extraction failed",sys)