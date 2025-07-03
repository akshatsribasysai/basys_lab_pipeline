import os
import json
#########################################
# Define API Keys
landing_ai_api_key = ""
google_api_key = ""

os.environ["VISION_AGENT_API_KEY"]=landing_ai_api_key

#########################################
# Define JSON Schemas
from json_schemas import *
schemas = [Patient, Visit, VisitNotes, Diagnosis, Symptom, Medication, VitalSigns, LabResult, ImagingStudy, ProcedureTreatment]

#########################################
# Generate Prompt
from json_prompt_gen import PromptGen
patient_id = 12434
PromptGenerator = PromptGen()
json_prompt = PromptGenerator.generate_json_prompt(schemas, patient_id)
print(json.dumps(json_prompt, indent=2))

#########################################
# Scrape text
from scrape_doc import PDFScraper
pdf_filepath = r""
Scraper = PDFScraper()
scraped_text = Scraper.extract_text_from_pdf_landingai(pdf_filepath)

print(scraped_text)

# #########################################
# # # Alternate to Scraping
with open(r'C:\Users\banan\Documents\basys\samples\medrecord.txt', 'r', encoding='utf-8') as file:
    scraped_text = file.read()

# #########################################
# Analyze text
from analyze_doc import DocAnalyzer

Analyzer = DocAnalyzer(google_api_key)
chunks = Analyzer.chunk_text(scraped_text)
results = Analyzer.ask_questions_on_chunks(chunks, patient_id, json_prompt)
print(results)
results_json = json.dumps(results)
print(results_json)

# # Alternate to prompting Gemini
# with open('/samples/json_sample.txt', 'r', encoding='utf-8') as file:
#     results_json = file.read()

analyze_results_json = results_json

# # #########################################
# Save to SQL

from save_to_sql import SQLSaver
db_config = {
    'host': 'localhost',
    'database': 'medical_records',
    'user': 'root',
    'password': 'pwd'
}
