#################################import necessary libraries############################################################

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import json
from datetime import datetime

#my sql integration libraries
import mysql.connector
from mysql.connector import Error

# Set up environment variables
os.environ["VISION_AGENT_API_KEY"] = "NjBhaDZzczJzcm16bXk2N3lwcTg5OjJIMWtXWnNkN2k3RnMyQ1RNbGJuNXJzMFZPN3NlYjd6"

from agentic_doc.parse import parse
###########################################################################################################################

class Pipeline1:
    def __init__(self, google_api_key):
        self.google_api_key = google_api_key

    def extract_text_from_pdf_landingai(self, file_path: str):
        """Extract text from PDF using Landing AI"""
        try:
            if parse is None:
                raise ImportError("agentic_doc.parse not available")
                
            result = parse(file_path)
            if result and len(result) > 0:
                parsed_doc = result[0]
                print("Extracted content successfully")
                print(f"Content preview: {str(result)[:500]}...")
                return str(result)
            else:
                print("No content extracted from PDF")
                return ""
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

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

    def ask_questions_on_chunks(self, docs, question):
        """Ask questions on document chunks using Gemini"""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=self.google_api_key,
                temperature=0.1
            )
            
            chain = load_qa_chain(llm, chain_type="stuff")
            result = chain.run(input_documents=docs, question=question)
            return result
        except Exception as e:
            print(f"Error processing with Gemini: {e}")
            return None
        



#component to save extracted JSON data to MySQL database       
    def save_to_mysql(self, json_data, db_config):
        """Save extracted JSON data to MySQL database"""
        try:
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            
            # Parse JSON data
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            
            # Insert Patient
            patient_query = """
            INSERT INTO patients (medical_record_number, created_date, updated_date) 
            VALUES (%s, %s, %s)
            """
            patient_data = data['patient']
            cursor.execute(patient_query, (
                patient_data['medical_record_number'],
                patient_data['created_date'],
                patient_data.get('updated_date')
            ))
            patient_id = cursor.lastrowid
            
            # Insert Provider if exists
            if data.get('provider'):
                provider_query = """
                INSERT INTO providers (provider_name, npi_number, specialty, department_id, active_status) 
                VALUES (%s, %s, %s, %s, %s)
                """
                provider_data = data['provider']
                cursor.execute(provider_query, (
                    provider_data['provider_name'],
                    provider_data['npi_number'],
                    provider_data.get('specialty'),
                    provider_data.get('department_id'),
                    provider_data.get('active_status', True)
                ))
                provider_id = cursor.lastrowid
            
            # Insert Department if exists
            if data.get('department'):
                dept_query = """
                INSERT INTO departments (department_name, department_type, system_name) 
                VALUES (%s, %s, %s)
                """
                dept_data = data['department']
                cursor.execute(dept_query, (
                    dept_data['department_name'],
                    dept_data.get('department_type'),
                    dept_data.get('system_name')
                ))
                dept_id = cursor.lastrowid
            
            # Insert Visits
            for visit in data.get('visits', []):
                visit_query = """
                INSERT INTO visits (patient_id, visit_date, visit_type, department_id, primary_provider_id, discharge_date) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(visit_query, (
                    patient_id, visit['visit_date'], visit['visit_type'],
                    visit.get('department_id'), visit.get('primary_provider_id'), visit.get('discharge_date')
                ))
                visit_id = cursor.lastrowid
                
                # Insert Visit Notes
                if visit.get('full_note_text'):
                    note_query = """
                    INSERT INTO visit_notes (visit_id, note_date, note_type, full_note_text, chief_complaint, 
                    history_present_illness, review_of_systems, physical_exam, assessment, plan, author_provider_id,
                    extraction_confidence, extraction_method, extraction_timestamp) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(note_query, (
                        visit_id, visit.get('note_date'), visit.get('note_type'), visit.get('full_note_text'),
                        visit.get('chief_complaint'), visit.get('history_present_illness'), visit.get('review_of_systems'),
                        visit.get('physical_exam'), visit.get('assessment'), visit.get('plan'), visit.get('author_provider_id'),
                        visit.get('extraction_confidence'), visit.get('extraction_method'), visit.get('extraction_timestamp')
                    ))
            
            # Insert Diagnoses
            for diagnosis in data.get('diagnoses', []):
                diag_query = """
                INSERT INTO diagnoses (patient_id, visit_id, diagnosis_name, icd10_code, onset_date, resolution_date,
                is_chronic, is_active, severity, diagnosing_provider_id, diagnosis_source, diagnosis_context,
                confidence_score, updated_date) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(diag_query, (
                    patient_id, diagnosis.get('visit_id'), diagnosis['diagnosis_name'], diagnosis.get('icd10_code'),
                    diagnosis.get('onset_date'), diagnosis.get('resolution_date'), diagnosis.get('is_chronic', False),
                    diagnosis.get('is_active', True), diagnosis.get('severity'), diagnosis.get('diagnosing_provider_id'),
                    diagnosis.get('diagnosis_source'), diagnosis.get('diagnosis_context'), diagnosis.get('confidence_score'),
                    diagnosis.get('updated_date')
                ))
            
            # Similar inserts for other tables...
            # (medications, symptoms, vital_signs, lab_results, imaging_studies, procedures)
            
            connection.commit()
            print(f"Data successfully saved to MySQL database")
            
        except Error as e:
            print(f"Error saving to MySQL: {e}")
            if connection:
                connection.rollback()
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

def main():
    ############################# Configuration###########################################
    PDF_PATH = r"C:\codes\agentic_ai_basys\training_files_patient\1.pdf"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
    #######################################################################################
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        return
    
    pipeline1_obj = Pipeline1(GOOGLE_API_KEY)
    
    try:
        # Step 1: Extract text from PDF
        print("Step 1: Extracting text from PDF...")
        extracted_text = pipeline1_obj.extract_text_from_pdf_landingai(PDF_PATH)
        
        if not extracted_text:
            print("No text extracted. Exiting.")
            return
        
        # Step 2: Split into chunks
        print("Step 2: Splitting text into chunks...")
        chunks = pipeline1_obj.chunk_text(extracted_text)
        
        # Step 3: Extract medical data in structured JSON format
        print("Step 3: Extracting medical data using Gemini...")
        
        question = """
        Extract all medical information from this document and format it as a JSON object matching this exact structure. 
        Use null for missing values. All dates should be in ISO format (YYYY-MM-DDTHH:MM:SS).
        
        Return ONLY the JSON object with this structure:
        {
            "patient": {
                "patient_id": 1,
                "medical_record_number": "extracted_or_generated_mrn",
                "created_date": "current_datetime_iso",
                "updated_date": null
            },
            "provider": {
                "provider_id": 1,
                "provider_name": "doctor_name_if_found",
                "npi_number": "npi_if_found_or_generated",
                "specialty": "specialty_if_found",
                "department_id": null,
                "active_status": true
            },
            "department": {
                "department_id": 1,
                "department_name": "department_if_found",
                "department_type": "type_if_found",
                "system_name": null
            },
            "visits": [
                {
                    "visit_id": 1,
                    "patient_id": 1,
                    "visit_date": "visit_date_iso",
                    "visit_type": "visit_type_extracted",
                    "department_id": 1,
                    "primary_provider_id": 1,
                    "discharge_date": null,
                    "note_id": 1,
                    "note_date": "note_date_iso",
                    "note_type": "note_type",
                    "full_note_text": "complete_note_text",
                    "chief_complaint": "chief_complaint_text",
                    "history_present_illness": "hpi_text",
                    "review_of_systems": "ros_text",
                    "physical_exam": "pe_text",
                    "assessment": "assessment_text",
                    "plan": "plan_text",
                    "author_provider_id": 1,
                    "extraction_confidence": 0.95,
                    "extraction_method": "gemini_langchain",
                    "extraction_timestamp": "current_datetime_iso"
                }
            ],
            "diagnoses": [
                {
                    "diagnosis_id": 1,
                    "patient_id": 1,
                    "visit_id": 1,
                    "diagnosis_name": "diagnosis_name",
                    "icd10_code": "icd10_code_if_found",
                    "onset_date": "onset_date_iso_if_found",
                    "resolution_date": null,
                    "is_chronic": false,
                    "is_active": true,
                    "severity": "severity_if_mentioned",
                    "diagnosing_provider_id": 1,
                    "diagnosis_source": "clinical_notes",
                    "diagnosis_context": "context_if_available",
                    "confidence_score": 0.9,
                    "updated_date": null
                }
            ],
            "symptoms": [
                {
                    "symptom_id": 1,
                    "patient_id": 1,
                    "visit_id": 1,
                    "symptom_name": "symptom_name",
                    "onset_date": "onset_date_if_found",
                    "duration": "duration_if_mentioned",
                    "frequency": "frequency_if_mentioned",
                    "severity": "severity_if_mentioned",
                    "symptom_description": "detailed_description",
                    "alleviating_factors": "factors_that_help",
                    "aggravating_factors": "factors_that_worsen",
                    "reported_date": "when_reported_iso",
                    "resolution_date": null
                }
            ],
            "medications": [
                {
                    "medication_id": 1,
                    "patient_id": 1,
                    "visit_id": 1,
                    "medication_name": "medication_name",
                    "generic_name": "generic_name_if_available",
                    "rxnorm_code": null,
                    "dose": "dose_amount",
                    "dose_unit": "mg_or_unit",
                    "frequency": "frequency_description",
                    "route": "route_of_administration",
                    "start_date": "start_date_if_available",
                    "end_date": null,
                    "discontinuation_reason": null,
                    "is_active": true,
                    "is_prn": false,
                    "prescribing_provider_id": 1,
                    "sig_text": "sig_instructions",
                    "patient_instructions": "patient_instructions",
                    "updated_date": null
                }
            ],
            "vital_signs": [
                {
                    "vital_sign_id": 1,
                    "patient_id": 1,
                    "visit_id": 1,
                    "measurement_datetime": "measurement_datetime_iso",
                    "weight_kg": null,
                    "height_cm": null,
                    "bmi": null,
                    "pulse_bpm": null,
                    "blood_pressure_systolic": null,
                    "blood_pressure_diastolic": null,
                    "temperature_celsius": null,
                    "respiratory_rate": null,
                    "oxygen_saturation_percent": null,
                    "pain_scale": null,
                    "additional_vitals": "any_other_vitals",
                    "measurement_context": "context_of_measurement",
                    "measured_by_id": 1
                }
            ],
            "lab_results": [
                {
                    "lab_result_id": 1,
                    "patient_id": 1,
                    "visit_id": 1,
                    "lab_name": "lab_name",
                    "test_name": "specific_test_name",
                    "loinc_code": null,
                    "result_value": "result_as_string",
                    "result_numeric": null,
                    "unit_of_measurement": "unit",
                    "reference_range_low": null,
                    "reference_range_high": null,
                    "reference_range_text": "reference_range_text",
                    "abnormality_flag": false,
                    "abnormality_type": null,
                    "collection_datetime": "collection_datetime_iso",
                    "result_datetime": "result_datetime_iso",
                    "ordering_provider_id": 1,
                    "result_status": "final",
                    "clinical_significance": "significance_if_noted"
                }
            ],
            "imaging_studies": [
                {
                    "imaging_id": 1,
                    "patient_id": 1,
                    "visit_id": 1,
                    "imaging_type": "xray_ct_mri_etc",
                    "modality": "specific_modality",
                    "body_region": "body_region_studied",
                    "study_datetime": "study_datetime_iso",
                    "ordering_provider_id": 1,
                    "radiologist_id": null,
                    "indication": "reason_for_study",
                    "technique": "technique_used",
                    "comparison": "comparison_studies",
                    "findings": "findings_text",
                    "impression": "impression_text",
                    "key_findings": "key_findings_summary",
                    "report_status": "final",
                    "critical_findings": false
                }
            ],
            "procedures": [
                {
                    "procedure_id": 1,
                    "patient_id": 1,
                    "visit_id": 1,
                    "procedure_name": "procedure_name",
                    "procedure_type": "type_of_procedure", 
                    "cpt_code": null,
                    "procedure_date": "procedure_date_iso",
                    "duration_minutes": null,
                    "outcome": "outcome_description",
                    "outcome_details": "detailed_outcome",
                    "complications": null,
                    "primary_provider_id": 1,
                    "therapy_type": null,
                    "sessions_completed": null,
                    "sessions_planned": null
                }
            ]
        }
        
        Extract all available information from the medical document. If information is not available, use null. 
        Ensure all arrays contain at least one object if any relevant data is found, otherwise use empty arrays [].
        """
        
        answer = pipeline1_obj.ask_questions_on_chunks(chunks, question)
        
        if answer:
            print("\n" + "="*50)
            print("EXTRACTED MEDICAL RECORD (JSON):")
            print("="*50)
            
            try:
                # Clean the response to extract JSON
                json_start = answer.find('{')
                json_end = answer.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    json_str = answer[json_start:json_end]
                    
                    # Parse and validate JSON
                    parsed_json = json.loads(json_str)
                    
                    # Pretty print the JSON
                    formatted_json = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    print(formatted_json)
                    
                    # Optionally save to file
                    output_file = "extracted_medical_record.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(formatted_json)
                    print(f"\nJSON saved to: {output_file}")
                    
                else:
                    print("No valid JSON found in response.")
                    print("Raw response:")
                    print(answer)
                    
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print("Raw response:")
                print(answer)
        else:
            print("No response received from Gemini")


        # Add database configuration
        db_config = {
            'host': 'localhost',
            'database': 'medical_records',
            'user': 'root',
            'password': '27ramome'
        }

        # Save to database
        if parsed_json:
            pipeline1_obj.save_to_mysql(parsed_json, db_config)
            
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()