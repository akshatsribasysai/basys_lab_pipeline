
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
Landing_api_key=os.getenv("VISION_AGENT_API_KEY", "LandingAI_API_KEY_HERE")
os.environ["VISION_AGENT_API_KEY"] = Landing_api_key


from agentic_doc.parse import parse

#for pydantic model
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from datetime import date
###########################################################################################################################








# Define Pydantic models for each table in the schema - FIXED MODELS
class Patient(BaseModel):
    """Patient table schema"""
    patient_id: int = Field(..., description="Primary key")
    medical_record_number: str = Field(..., description="Unique medical record number")
    created_date: datetime = Field(default_factory=datetime.now)
    updated_date: Optional[datetime] = None

class Visit(BaseModel):
    """Visit table schema"""
    visit_date: str  # Changed to string to handle various date formats
    visit_type: str
    department_name: str
    primary_provider_name: str
    discharge_date: Optional[str] = None  # Changed to optional string

class VisitNotes(BaseModel):
    """Visit Notes table schema"""
    visit_id: Optional[int] = None
    note_date: Optional[str] = None  # Changed to string
    note_type: Optional[str] = None
    full_note_text: Optional[str] = None
    chief_complaint: Optional[str] = None
    history_present_illness: Optional[str] = None
    review_of_systems: Optional[str] = None
    physical_exam: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    author_provider_id: Optional[int] = None
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    extraction_method: Optional[str] = None
    extraction_timestamp: Optional[str] = None  # Changed to string

class Diagnosis(BaseModel):
    """Diagnosis table schema - FIXED"""
    patient_id: Optional[int] = None  # Made optional to prevent validation errors
    visit_id: Optional[int] = None
    diagnosis_name: str
    icd10_code: Optional[str] = None
    onset_date: Optional[str] = None  # Changed to string
    resolution_date: Optional[str] = None  # Changed to string
    is_chronic: Optional[bool] = Field(default=False)
    is_active: Optional[bool] = Field(default=True)
    severity: Optional[str] = None
    diagnosing_provider_id: Optional[int] = None
    diagnosis_source: Optional[str] = None
    diagnosis_context: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    updated_date: Optional[str] = None  # Changed to string

class Symptom(BaseModel):
    """Symptom table schema - FIXED"""
    patient_id: Optional[int] = None  # Made optional
    visit_id: Optional[int] = None
    symptom_name: str
    onset_date: Optional[str] = None  # Changed to string
    duration: Optional[str] = None
    frequency: Optional[str] = None
    severity: Optional[str] = None
    symptom_description: Optional[str] = None
    alleviating_factors: Optional[str] = None
    aggravating_factors: Optional[str] = None
    reported_date: Optional[str] = None  # Changed to string
    resolution_date: Optional[str] = None  # Changed to string

class Medication(BaseModel):
    """Medication table schema - FIXED"""
    patient_id: Optional[int] = None  # Made optional
    visit_id: Optional[int] = None
    medication_name: str
    generic_name: Optional[str] = None
    rxnorm_code: Optional[str] = None
    dose: Optional[str] = None
    dose_unit: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    start_date: Optional[str] = None  # Changed to string
    end_date: Optional[str] = None  # Changed to string
    discontinuation_reason: Optional[str] = None
    is_active: Optional[bool] = Field(default=True)
    is_prn: Optional[bool] = Field(default=False)
    prescribing_provider_id: Optional[int] = None
    sig_text: Optional[str] = None
    patient_instructions: Optional[str] = None
    updated_date: Optional[str] = None  # Changed to string

class VitalSigns(BaseModel):
    """Vital signs table schema - FIXED"""
    patient_id: Optional[int] = None  # Made optional
    visit_id: Optional[int] = None
    measurement_datetime: str  # Changed to string
    weight_kg: Optional[float] = Field(None, ge=0)
    height_cm: Optional[float] = Field(None, ge=0)
    bmi: Optional[float] = Field(None, ge=0)
    pulse_bpm: Optional[int] = Field(None, ge=0)
    blood_pressure_systolic: Optional[int] = Field(None, ge=0)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=0)
    temperature_celsius: Optional[float] = Field(None, ge=0)
    respiratory_rate: Optional[int] = Field(None, ge=0)
    oxygen_saturation_percent: Optional[int] = Field(None, ge=0, le=100)
    pain_scale: Optional[int] = Field(None, ge=0, le=10)
    additional_vitals: Optional[str] = None
    measurement_context: Optional[str] = None
    measured_by_id: Optional[int] = None

class LabResult(BaseModel):
    """Lab result table schema - FIXED"""
    patient_id: Optional[int] = None  # Made optional
    visit_id: Optional[int] = None
    lab_name: str
    test_name: str
    loinc_code: Optional[str] = None
    result_value: str
    result_numeric: Optional[float] = None
    unit_of_measurement: Optional[str] = None
    reference_range_low: Optional[float] = None
    reference_range_high: Optional[float] = None
    reference_range_text: Optional[str] = None
    abnormality_flag: Optional[bool] = Field(default=False)
    abnormality_type: Optional[str] = None
    collection_datetime: Optional[str] = None  # Changed to string
    result_datetime: Optional[str] = None  # Changed to string
    ordering_provider_id: Optional[int] = None
    result_status: Optional[str] = None
    clinical_significance: Optional[str] = None

class ImagingStudy(BaseModel):
    """Imaging study table schema - FIXED"""
    patient_id: Optional[int] = None  # Made optional
    visit_id: Optional[int] = None
    imaging_type: str
    modality: Optional[str] = None
    body_region: Optional[str] = None
    study_datetime: str  # Changed to string
    ordering_provider_id: Optional[int] = None
    radiologist_id: Optional[int] = None
    indication: Optional[str] = None
    technique: Optional[str] = None
    comparison: Optional[str] = None
    findings: Optional[str] = None
    impression: Optional[str] = None
    key_findings: Optional[str] = None
    report_status: Optional[str] = None
    critical_findings: Optional[bool] = Field(default=False)

class ProcedureTreatment(BaseModel):
    """Procedure treatment table schema - FIXED"""
    patient_id: Optional[int] = None  # Made optional
    visit_id: Optional[int] = None
    procedure_name: str
    procedure_type: Optional[str] = None
    cpt_code: Optional[str] = None
    procedure_date: str  # Changed to string
    duration_minutes: Optional[int] = Field(None, ge=0)
    outcome: Optional[str] = None
    outcome_details: Optional[str] = None
    complications: Optional[str] = None
    primary_provider_id: Optional[int] = None
    therapy_type: Optional[str] = None
    sessions_completed: Optional[int] = Field(None, ge=0)
    sessions_planned: Optional[int] = Field(None, ge=0)

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

    def ask_questions_on_chunks(self, docs, question, patient_id):
        """Ask questions on document chunks using Gemini with Pydantic validation"""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=self.google_api_key,
                temperature=0.1
            )
            
            # Enhanced prompt to prevent hallucination
            enhanced_question = f"""
            {question}
            
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
            {{
                "visits": [{{
                    "visit_date": "string (date)",
                    "visit_type": "string",
                    "department_name": "string", 
                    "primary_provider_name": "string",
                    "discharge_date": "string or null"
                }}],
                "diagnoses": [{{
                    "patient_id": {patient_id},
                    "diagnosis_name": "string",
                    "icd10_code": "string or null",
                    "is_chronic": false,
                    "is_active": true,
                    "severity": "string or null"
                }}],
                "medications": [{{
                    "patient_id": {patient_id},
                    "medication_name": "string",
                    "dose": "string or null",
                    "frequency": "string or null",
                    "route": "string or null",
                    "is_active": true
                }}],
                "symptoms": [{{
                    "patient_id": {patient_id},
                    "symptom_name": "string",
                    "severity": "string or null",
                    "duration": "string or null"
                }}],
                "vital_signs": [{{
                    "patient_id": {patient_id},
                    "measurement_datetime": "string (date)",
                    "weight_kg": number or null,
                    "height_cm": number or null,
                    "pulse_bpm": number or null,
                    "blood_pressure_systolic": number or null,
                    "blood_pressure_diastolic": number or null,
                    "temperature_celsius": number or null
                }}]
            }}

            Use these EXACT field names. Always include patient_id with value {patient_id} where required.
            """
            
            from langchain.chains import StuffDocumentsChain
            from langchain.chains.llm import LLMChain
            from langchain.prompts import PromptTemplate

            # Create prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end.

            {context}

            Question: {question}
            Answer:"""

            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

            result = stuff_chain.invoke({"input_documents": docs, "question": enhanced_question})
            result = result["output_text"]

            
            # Extract and validate JSON with Pydantic
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_str = result[json_start:json_end]
                try:
                    raw_data = json.loads(json_str)
                    
                    # Fill in patient_id for all records that need it
                    if raw_data.get('diagnoses'):
                        for diagnosis in raw_data['diagnoses']:
                            if diagnosis.get('patient_id') is None:
                                diagnosis['patient_id'] = patient_id
                    
                    if raw_data.get('medications'):
                        for medication in raw_data['medications']:
                            if medication.get('patient_id') is None:
                                medication['patient_id'] = patient_id
                    
                    if raw_data.get('symptoms'):
                        for symptom in raw_data['symptoms']:
                            if symptom.get('patient_id') is None:
                                symptom['patient_id'] = patient_id
                    
                    if raw_data.get('vital_signs'):
                        for vital in raw_data['vital_signs']:
                            if vital.get('patient_id') is None:
                                vital['patient_id'] = patient_id
                    
                    # Add patient_id to other records as needed
                    for record_type in ['lab_results', 'imaging_studies', 'procedures']:
                        if raw_data.get(record_type):
                            for record in raw_data[record_type]:
                                if record.get('patient_id') is None:
                                    record['patient_id'] = patient_id
                    
                    # Validate each section with Pydantic models - only if data exists
                    validated_data = {}
                    
                    try:
                        if raw_data.get('visits'):
                            validated_data['visits'] = [Visit(**v) for v in raw_data['visits']]
                            print(f"✓ Validated {len(validated_data['visits'])} visits")
                        
                        if raw_data.get('visit_notes'):
                            validated_data['visit_notes'] = [VisitNotes(**vn) for vn in raw_data['visit_notes']]
                            print(f"✓ Validated {len(validated_data['visit_notes'])} visit notes")
                        
                        if raw_data.get('diagnoses'):
                            validated_data['diagnoses'] = [Diagnosis(**d) for d in raw_data['diagnoses']]
                            print(f"✓ Validated {len(validated_data['diagnoses'])} diagnoses")
                        
                        if raw_data.get('symptoms'):
                            validated_data['symptoms'] = [Symptom(**s) for s in raw_data['symptoms']]
                            print(f"✓ Validated {len(validated_data['symptoms'])} symptoms")
                        
                        if raw_data.get('medications'):
                            validated_data['medications'] = [Medication(**m) for m in raw_data['medications']]
                            print(f"✓ Validated {len(validated_data['medications'])} medications")
                        
                        if raw_data.get('vital_signs'):
                            validated_data['vital_signs'] = [VitalSigns(**vs) for vs in raw_data['vital_signs']]
                            print(f"✓ Validated {len(validated_data['vital_signs'])} vital signs")
                        
                        if raw_data.get('lab_results'):
                            validated_data['lab_results'] = [LabResult(**lr) for lr in raw_data['lab_results']]
                            print(f"✓ Validated {len(validated_data['lab_results'])} lab results")
                        
                        if raw_data.get('imaging_studies'):
                            validated_data['imaging_studies'] = [ImagingStudy(**img) for img in raw_data['imaging_studies']]
                            print(f"✓ Validated {len(validated_data['imaging_studies'])} imaging studies")
                        
                        if raw_data.get('procedures'):
                            validated_data['procedures'] = [ProcedureTreatment(**p) for p in raw_data['procedures']]
                            print(f"✓ Validated {len(validated_data['procedures'])} procedures")
                        
                        print("✓ All data validated successfully with Pydantic")
                        return raw_data
                        
                    except ValidationError as ve:
                        print(f"Pydantic validation error: {ve}")
                        print("Raw data that failed validation:")
                        print(json.dumps(raw_data, indent=2, default=str))
                        # Return raw data even if validation fails, so we can see what was extracted
                        return raw_data
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    print(f"Raw response: {result}")
                    return None
            else:
                print("No valid JSON found in response")
                print(f"Raw response: {result}")
                return None
                
        except Exception as e:
            print(f"Error processing with Gemini: {e}")
            return None

    def save_to_mysql(self, json_data, db_config, patient_id):
        """Save extracted JSON data to MySQL database"""
        connection = None
        cursor = None
        
        try:
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            
            # Parse JSON data
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            
            # Validate patient exists if not then creates new patient
            patient_check_query = "SELECT patient_id FROM patients WHERE patient_id = %s"
            cursor.execute(patient_check_query, (patient_id,))
            if not cursor.fetchone():
                # Create new patient
                create_patient_query = """
                INSERT INTO patients (patient_id, medical_record_number, created_date) 
                VALUES (%s, %s, %s)
                """
                medical_record_number = f"MRN{patient_id:06d}"  # Generate MRN like MRN000001
                cursor.execute(create_patient_query, (patient_id, medical_record_number, datetime.now()))
                print(f"✓ Created new patient with ID {patient_id} and MRN {medical_record_number}")
            
            # Insert Visits - only if data exists
            visit_ids = []
            if data.get('visits'):
                for visit_data in data['visits']:
                    visit_query = """
                    INSERT INTO visits (patient_id, visit_date, visit_type, department_name, primary_provider_name, discharge_date) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    # Convert date string to datetime if needed
                    visit_date = visit_data['visit_date']
                    if isinstance(visit_date, str):
                        try:
                            # Try to parse common date formats
                            if '/' in visit_date:
                                visit_date = datetime.strptime(visit_date, '%m/%d/%Y')
                            elif '-' in visit_date:
                                visit_date = datetime.strptime(visit_date, '%Y-%m-%d')
                        except:
                            visit_date = datetime.now()  # fallback
                    
                    cursor.execute(visit_query, (
                        patient_id, 
                        visit_date, 
                        visit_data['visit_type'],
                        visit_data['department_name'], 
                        visit_data['primary_provider_name'], 
                        visit_data.get('discharge_date')
                    ))
                    visit_ids.append(cursor.lastrowid)
                    print(f"✓ Inserted visit: {visit_data['visit_type']} on {visit_date}")
            
            # Insert Diagnoses - only if data exists
            if data.get('diagnoses'):
                for diag_data in data['diagnoses']:
                    diag_query = """
                    INSERT INTO diagnoses (patient_id, visit_id, diagnosis_name, icd10_code, onset_date, resolution_date,
                    is_chronic, is_active, severity, diagnosis_source, diagnosis_context, confidence_score) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(diag_query, (
                        diag_data.get('patient_id', patient_id),
                        diag_data.get('visit_id'),
                        diag_data['diagnosis_name'],
                        diag_data.get('icd10_code'),
                        diag_data.get('onset_date'),
                        diag_data.get('resolution_date'),
                        diag_data.get('is_chronic', False),
                        diag_data.get('is_active', True),
                        diag_data.get('severity'),
                        diag_data.get('diagnosis_source'),
                        diag_data.get('diagnosis_context'),
                        diag_data.get('confidence_score')
                    ))
                    print(f"✓ Inserted diagnosis: {diag_data['diagnosis_name']}")
            
            # Insert Medications - only if data exists
            if data.get('medications'):
                for med_data in data['medications']:
                    med_query = """
                    INSERT INTO medications (patient_id, visit_id, medication_name, generic_name, dose, dose_unit,
                    frequency, route, start_date, end_date, is_active, is_prn, sig_text, patient_instructions) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(med_query, (
                        med_data.get('patient_id', patient_id),
                        med_data.get('visit_id'),
                        med_data['medication_name'],
                        med_data.get('generic_name'),
                        med_data.get('dose'),
                        med_data.get('dose_unit'),
                        med_data.get('frequency'),
                        med_data.get('route'),
                        med_data.get('start_date'),
                        med_data.get('end_date'),
                        med_data.get('is_active', True),
                        med_data.get('is_prn', False),
                        med_data.get('sig_text'),
                        med_data.get('patient_instructions')
                    ))
                    print(f"✓ Inserted medication: {med_data['medication_name']}")
            
            # Insert Symptoms - only if data exists
            if data.get('symptoms'):
                for symptom_data in data['symptoms']:
                    symptom_query = """
                    INSERT INTO symptoms (patient_id, visit_id, symptom_name, onset_date, duration, frequency,
                    severity, symptom_description, alleviating_factors, aggravating_factors, reported_date, resolution_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(symptom_query, (
                        symptom_data.get('patient_id', patient_id),
                        symptom_data.get('visit_id'),
                        symptom_data['symptom_name'],
                        symptom_data.get('onset_date'),
                        symptom_data.get('duration'),
                        symptom_data.get('frequency'),
                        symptom_data.get('severity'),
                        symptom_data.get('symptom_description'),
                        symptom_data.get('alleviating_factors'),
                        symptom_data.get('aggravating_factors'),
                        symptom_data.get('reported_date'),
                        symptom_data.get('resolution_date')
                    ))
                    print(f"✓ Inserted symptom: {symptom_data['symptom_name']}")
            
            # Insert Vital Signs - only if data exists
            if data.get('vital_signs'):
                for vs_data in data['vital_signs']:
                    vs_query = """
                    INSERT INTO vital_signs (patient_id, visit_id, measurement_datetime, weight_kg, height_cm, bmi,
                    pulse_bpm, blood_pressure_systolic, blood_pressure_diastolic, temperature_celsius, 
                    respiratory_rate, oxygen_saturation_percent, pain_scale) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    # Handle measurement_datetime
                    measurement_dt = vs_data.get('measurement_datetime')
                    if isinstance(measurement_dt, str):
                        try:
                            if '/' in measurement_dt:
                                measurement_dt = datetime.strptime(measurement_dt, '%m/%d/%Y')
                            elif '-' in measurement_dt:
                                measurement_dt = datetime.strptime(measurement_dt, '%Y-%m-%d')
                        except:
                            measurement_dt = datetime.now()
                    
                    cursor.execute(vs_query, (
                        vs_data.get('patient_id', patient_id), 
                        vs_data.get('visit_id'), 
                        measurement_dt,
                        vs_data.get('weight_kg'), 
                        vs_data.get('height_cm'), 
                        vs_data.get('bmi'),
                        vs_data.get('pulse_bpm'), 
                        vs_data.get('blood_pressure_systolic'), 
                        vs_data.get('blood_pressure_diastolic'), 
                        vs_data.get('temperature_celsius'),
                        vs_data.get('respiratory_rate'), 
                        vs_data.get('oxygen_saturation_percent'),
                        vs_data.get('pain_scale')
                    ))
                    print(f"✓ Inserted vital signs for {measurement_dt}")
            
            connection.commit()
            print(f"✓ Data successfully saved to MySQL database")
            
        except Error as e:
            print(f"Error saving to MySQL: {e}")
            if connection:
                connection.rollback()
        finally:
            if connection and connection.is_connected():
                if cursor:
                    cursor.close()
                connection.close()

def main():
    ############################# Configuration###########################################
    PDF_PATH = r"C:\codes\agentic_ai_basys\training_files_patient\1.pdf"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
    
    # Get patient ID from user
    patient_id = input("Enter Patient ID: ").strip()
    if not patient_id or not patient_id.isdigit():
        print("Error: Please enter a valid numeric Patient ID")
        return
    patient_id = int(patient_id)
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
        
        question = "Extract medical information and return as valid JSON matching the expected schema structure."
        
        answer = pipeline1_obj.ask_questions_on_chunks(chunks, question, patient_id)
        
        if answer:
            print("\n" + "="*50)
            print("EXTRACTED MEDICAL RECORD (JSON - Pydantic Validated):")
            print("="*50)
            
            # Pretty print the validated JSON
            formatted_json = json.dumps(answer, indent=2, ensure_ascii=False, default=str)
            print(formatted_json)
            
            # Save to file
            output_file = "extracted_medical_record.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"\n✓ Validated JSON saved to: {output_file}")
            
            # Add database configuration
            db_config = {
                'host': 'localhost',
                'database': 'medical_records',
                'user': 'root',
                'password': '27ramome'
            }

            # Save to database
            pipeline1_obj.save_to_mysql(answer, db_config, patient_id)
        else:
            print("No response received from Gemini")
            
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()