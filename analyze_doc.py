from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import os

class DocAnalyzer:
    def __init__ (self, API_key):
        self.google_api_key = API_key

    def chunk_text(self, text, chunk_size=1000, chunk_overlap=100):
        """Split text into chunks for processing"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            documents = splitter.create_documents([text])
            print(f"Created {len(documents)} chunks")
            return documents
        except Exception as e:
            print(f"Error chunking text: {e}")
            return [Document(page_content=text)]
    
    def ask_questions_on_chunks(self, docs, patient_id, json_prompt):
        """Ask questions on document chunks using Gemini with Pydantic validation"""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.1
            )

            prompt_template = """
            Extract medical information and return as valid JSON matching the expected schema structure.

            IMPORTANT INSTRUCTIONS:
            1. Only extract information that is explicitly present in the document
            2. Do not create, invent, or hallucinate any medical data
            3. If a section/table has no information in the document, return an empty array []
            4. If specific fields are not mentioned, leave them as null/None
            5. Be conservative - only include data you can clearly identify from the text
            6. Return valid JSON format only
            7. For patient_id field, use the provided patient_id: {patient_id}
            8. Use string format for all dates (e.g., "2013-12-30" or "12/30/2013")

            Expected JSON structure with exact field names:
            {json_prompt}

            Use these EXACT field names. Always include patient_id with value {patient_id} where required. 

            Context:
            {context}
            """

            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "patient_id","json_prompt"])
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

            result = stuff_chain.invoke({
                "input_documents": docs,
                "patient_id": patient_id,
                "json_prompt": json_prompt
            })
            result = result["output_text"]

            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = result[json_start:json_end]

                return json_str
            else:
                raise Exception

        except Exception as e:
            print(f"Error processing with Gemini: {e}")
            return None