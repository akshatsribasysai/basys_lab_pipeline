import os
import json

#########################################
# Define API Keys
landing_ai_api_key = os.environ.get("VISION_AGENT_API_KEY", "none")
google_api_key = os.environ.get("GOOGLE_API_KEY", "none")

os.environ["VISION_AGENT_API_KEY"]=landing_ai_api_key

#########################################
# Define JSON Schemas
from json_schemas import *
schemas = [Patient, Visit, VisitNotes, Diagnosis, Symptom, Medication, VitalSigns, LabResult, ImagingStudy, ProcedureTreatment]

#########################################
# Generate Prompt
from json_prompt_gen import PromptGen



patient_id = int(input("Enter Patient ID: "))
PromptGenerator = PromptGen()
json_prompt = PromptGenerator.generate_json_prompt(schemas, patient_id)
print(json.dumps(json_prompt, indent=2))

#########################################
# Scrape text
from scrape_doc import PDFScraper
pdf_filepath = r"C:\codes\agentic_ai_basys\training_files_patient\3.pdf"
Scraper = PDFScraper()
scraped_text = Scraper.extract_text_from_pdf_landingai(pdf_filepath)

print(scraped_text)

# #########################################
# # # Alternate to Scraping
""" with open(r'C:\codes\agentic_ai_basys\new_pydantic_pipeline\medrecord.txt', 'r', encoding='utf-8') as file:
    scraped_text = file.read() """

# #########################################
# Analyze text
from analyze_doc import DocAnalyzer

Analyzer = DocAnalyzer(google_api_key)
chunks = Analyzer.chunk_text(scraped_text)
results = Analyzer.ask_questions_on_chunks(chunks, patient_id, json_prompt)

results_json = results
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
    'password': '27ramome'
}



# Create SQLSaver instance
saver = SQLSaver()
print("=== DEBUG: Checking JSON data ===")
print(f"Type of analyze_results_json: {type(analyze_results_json)}")
print(f"First 500 characters: {analyze_results_json}")
with open('test_test_1.json', 'w+', encoding='utf-8') as f:
    f.write(analyze_results_json)
print("=== END DEBUG ===")

# Save the analyzed results to MySQL database
saver.save_to_mysql(analyze_results_json, db_config, patient_id)