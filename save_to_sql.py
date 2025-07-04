from json_schemas import *
import json
import mysql.connector
from datetime import datetime
import hashlib
import random

class SQLSaver:

    def generate_visit_hash(self, patient_id, visit_date, visit_type, department_name, primary_provider_name):
        """Generate a unique hash for a visit based on key identifying information"""
        # Create a string combining key visit identifiers
        visit_key = f"{patient_id}_{visit_date}_{visit_type}_{department_name}_{primary_provider_name}"
        # Generate SHA256 hash and take first 16 characters for reasonable uniqueness
        return hashlib.sha256(visit_key.encode()).hexdigest()[:16]

    def find_or_create_visit(self, cursor, patient_id, visit_data):
        """Find existing visit or create new one, return visit_id"""
        
        # Normalize visit date
        visit_date = visit_data.get('visit_date', datetime.now().date())
        if isinstance(visit_date, str):
            try:
                if '/' in visit_date:
                    visit_date = datetime.strptime(visit_date, '%m/%d/%Y').date()
                elif '-' in visit_date:
                    visit_date = datetime.strptime(visit_date, '%Y-%m-%d').date()
            except:
                visit_date = datetime.now().date()
        
        # Extract department and provider info
        department_name = "Unknown"
        primary_provider_name = "Unknown"
        
        if 'department' in visit_data and visit_data['department']:
            department_name = visit_data['department'].get('department_name', 'Unknown')
        
        if 'primary_provider' in visit_data and visit_data['primary_provider']:
            primary_provider_name = visit_data['primary_provider'].get('provider_name', 'Unknown')
        
        # Generate visit hash for uniqueness check
        visit_hash = self.generate_visit_hash(
            patient_id, 
            visit_date, 
            visit_data.get('visit_type', 'Unknown'),
            department_name, 
            primary_provider_name
        )
        
        # First, check if this exact visit already exists using our hash
        check_query = """
        SELECT visit_id FROM visits 
        WHERE patient_id = %s 
        AND visit_date = %s 
        AND visit_type = %s 
        AND department_name = %s 
        AND primary_provider_name = %s
        """
        
        cursor.execute(check_query, (
            patient_id,
            visit_date,
            visit_data.get('visit_type', 'Unknown'),
            department_name,
            primary_provider_name
        ))
        
        existing_visit = cursor.fetchone()
        
        if existing_visit:
            print(f"✓ Found existing visit with ID: {existing_visit[0]}")
            return existing_visit[0]
        
        # If no existing visit found, create new one
        # Use the visit_id from the data if available, otherwise generate one
        visit_id = self.generate_unique_visit_id(cursor)
        
        visit_query = """
        INSERT INTO visits (visit_id, patient_id, visit_date, visit_type, department_name, primary_provider_name, discharge_date, created_date) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(visit_query, (
            visit_id,
            patient_id, 
            visit_date, 
            visit_data.get('visit_type', 'Unknown'),
            department_name, 
            primary_provider_name, 
            self.parse_date(visit_data.get('discharge_date')),
            datetime.now()
        ))
        
        print(f"✓ Created new visit with ID: {visit_id} for {visit_data.get('visit_type', 'Unknown')} on {visit_date}")
        return visit_id

    def generate_unique_visit_id(self, cursor):
        """Generate a unique visit ID that doesn't exist in the database"""
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Generate a random 8-digit ID
            visit_id = random.randint(10000000, 99999999)
            
            # Check if this ID already exists
            cursor.execute("SELECT visit_id FROM visits WHERE visit_id = %s", (visit_id,))
            if not cursor.fetchone():
                return visit_id
        
        # If we can't find a unique ID after max_attempts, use timestamp-based approach
        import time
        timestamp_id = int(time.time() * 1000) % 100000000  # Use last 8 digits of timestamp
        
        # Ensure it's still unique
        cursor.execute("SELECT visit_id FROM visits WHERE visit_id = %s", (timestamp_id,))
        if not cursor.fetchone():
            return timestamp_id
        
        # Final fallback: find the max ID and increment
        cursor.execute("SELECT MAX(visit_id) FROM visits")
        max_id = cursor.fetchone()[0]
        return (max_id + 1) if max_id else 10000000

    def isolate_jsons(self, json_str, patient_id):
        """Extract and validate JSON with Pydantic"""
        
        raw_data = json.loads(json_str)
        
        # Fill in patient_id for all records that need it
        record_types = ['diagnosis', 'medication', 'symptom', 'vital_signs', 'lab_results', 'imaging_studies', 'procedures']
        
        for record_type in record_types:
            if raw_data.get(record_type):
                for record in raw_data[record_type]:
                    if record.get('patient_id') is None:
                        record['patient_id'] = patient_id
        
        # Validate each section with Pydantic models - only if data exists
        validated_data = {}
        
        print(f"Processing data for patient {patient_id}")
        
        if raw_data.get('visit'):
            validated_data['visit'] = [Visit(**raw_data['visit'])]
            print(f"✓ Validated {len(validated_data['visit'])} visit")

        if raw_data.get('visitnotes'):
            validated_data['visitnotes'] = [VisitNotes(**vn) for vn in raw_data['visitnotes']]
            print(f"✓ Validated {len(validated_data['visitnotes'])} visit notes")

        if raw_data.get('diagnosis'):
            validated_data['diagnosis'] = [Diagnosis(**d) for d in raw_data['diagnosis']]
            print(f"✓ Validated {len(validated_data['diagnosis'])} diagnosis")

        if raw_data.get('symptom'):
            validated_data['symptom'] = [Symptom(**s) for s in raw_data['symptom']]
            print(f"✓ Validated {len(validated_data['symptom'])} symptom")

        if raw_data.get('medication'):
            validated_data['medication'] = [Medication(**m) for m in raw_data['medication']]
            print(f"✓ Validated {len(validated_data['medication'])} medication")

        if raw_data.get('vital_sign'):
            validated_data['vital_sign'] = [VitalSigns(**vs) for vs in raw_data['vital_sign']]
            print(f"✓ Validated {len(validated_data['vital_sign'])} vital sign")

        if raw_data.get('lab_result'):
            validated_data['lab_result'] = [LabResult(**lr) for lr in raw_data['lab_result']]
            print(f"✓ Validated {len(validated_data['lab_result'])} lab result")

        if raw_data.get('imaging_study'):
            validated_data['imaging_study'] = [ImagingStudy(**img) for img in raw_data['imaging_study']]
            print(f"✓ Validated {len(validated_data['imaging_study'])} imaging study")

        if raw_data.get('procedure'):
            validated_data['procedure'] = [ProcedureTreatment(**p) for p in raw_data['procedure']]
            print(f"✓ Validated {len(validated_data['procedure'])} procedure")
        
        print("✓ All data validated successfully")
        return raw_data
    



    def parse_date(self, date_value):
        """Parse date string into proper format for MySQL"""
        if isinstance(date_value, str):
            try:
                # Handle year-only format (e.g., '2003')
                if len(date_value) == 4 and date_value.isdigit():
                    return datetime.strptime(f"{date_value}-01-01", '%Y-%m-%d').date()
                # Handle existing formats
                elif '/' in date_value:
                    return datetime.strptime(date_value, '%m/%d/%Y').date()
                elif '-' in date_value:
                    return datetime.strptime(date_value, '%Y-%m-%d').date()
            except:
                return None
        return date_value
    




    def save_to_mysql(self, json_data, db_config, patient_id: int):
        """Save extracted JSON data to MySQL database with proper visit ID management"""
        connection = None
        cursor = None
        
        try:
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            
            # Parse JSON data if it's a string
            print(f"=== DEBUG: Checking JSON data ===")
            print(f"Type of json_data: {type(json_data)}")
            print(f"First 200 characters: {str(json_data)[:200]}")
            
            if isinstance(json_data, str):
                try:
                    data = json.loads(json_data)
                    print(f"✓ Successfully parsed JSON string")
                except json.JSONDecodeError as e:
                    print(f"✗ JSON parsing error: {e}")
                    print(f"Problematic JSON around position {e.pos}: {json_data[max(0, e.pos-50):e.pos+50]}")
                    raise
            else:
                data = json_data
                print(f"✓ Data is already a dictionary")
            
            # Verify data is now a dictionary
            if not isinstance(data, dict):
                raise ValueError(f"Expected dictionary, got {type(data)}")
                
            print(f"=== Processing data for patient {patient_id} ===")
            print(f"Available data sections: {list(data.keys())}")
            
            # Validate patient exists, create if not
            patient_check_query = "SELECT patient_id FROM patients WHERE patient_id = %s"
            cursor.execute(patient_check_query, (patient_id,))
            if not cursor.fetchone():
                # Create new patient
                create_patient_query = """
                INSERT INTO patients (patient_id, medical_record_number, created_date) 
                VALUES (%s, %s, %s)
                """
                medical_record_number = f"MRN{patient_id:06d}"
                cursor.execute(create_patient_query, (patient_id, medical_record_number, datetime.now()))
                print(f"✓ Created new patient with ID {patient_id} and MRN {medical_record_number}")
            
            # Handle visits - find existing or create new
            visit_ids = []
            if data.get('visit'):
                # Handle visit data (could be single dict or list)
                visit_data = data['visit']
                if isinstance(visit_data, list):
                    for visit in visit_data:
                        visit_id = self.find_or_create_visit(cursor, patient_id, visit)
                        visit_ids.append(visit_id)
                else:
                    visit_id = self.find_or_create_visit(cursor, patient_id, visit_data)
                    visit_ids.append(visit_id)
            
            # If no visit data provided, we need to create a default visit for orphaned records
            if not visit_ids and (data.get('diagnosis') or data.get('medication') or data.get('symptom') or data.get('vital_signs')):
                print("⚠ No visit data found, creating default visit for orphaned records")
                default_visit = {
                    'visit_date': datetime.now().date(),
                    'visit_type': 'Unknown',
                    'department_name': 'Unknown',
                    'primary_provider_name': 'Unknown'
                }
                visit_id = self.find_or_create_visit(cursor, patient_id, default_visit)
                visit_ids.append(visit_id)
            
            # Use the first visit_id for all related records
            primary_visit_id = visit_ids[0] if visit_ids else None
            
            # Insert Visit Notes
            if data.get('visitnotes'):
                note_data = data['visitnotes']
                    # Skip if note_data is a string instead of dict
                if isinstance(note_data, str):
                    print(f"⚠ Skipping string note data: {note_data[:100]}...")
                note_query = """
                    INSERT INTO visit_notes (patient_id, visit_id, note_date, note_type, full_note_text, 
                    chief_complaint, history_present_illness, review_of_systems, physical_exam, 
                    assessment, plan, created_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    # Parse note_date
                note_date = self.parse_date(note_data.get('note_date'))
                if isinstance(note_date, str):
                    try:
                        if '/' in note_date:
                                note_date = datetime.strptime(note_date, '%m/%d/%Y').date()
                        elif '-' in note_date:
                                note_date = datetime.strptime(note_date, '%Y-%m-%d').date()
                    except:
                            note_date = datetime.now().date()
                    
                cursor.execute(note_query, (
                        note_data.get('patient_id', patient_id),
                        primary_visit_id,
                        note_date,
                        note_data.get('note_type'),
                        note_data.get('full_note_text'),
                        note_data.get('chief_complaint'),
                        note_data.get('history_present_illness'),
                        note_data.get('review_of_systems'),
                        note_data.get('physical_exam'),
                        note_data.get('assessment'),
                        note_data.get('plan'),
                        datetime.now()
                    ))
                print(f"✓ Inserted visit note: {note_data.get('note_type', 'Unknown type')}")
            
            # Insert Diagnoses
            if data.get('diagnosis'):
                for diag_data in data['diagnosis']:
                    diag_query = """
                    INSERT INTO diagnoses (patient_id, visit_id, diagnosis_name, icd10_code, onset_date, resolution_date,
                    is_chronic, is_active, severity, diagnosis_source, diagnosis_context, confidence_score, created_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(diag_query, (
                        diag_data.get('patient_id', patient_id),
                        primary_visit_id,
                        diag_data['diagnosis_name'],
                        diag_data.get('icd10_code'),
                        self.parse_date(diag_data.get('onset_date')),
                        self.parse_date(diag_data.get('resolution_date')),
                        diag_data.get('is_chronic', False),
                        diag_data.get('is_active', True),
                        diag_data.get('severity'),
                        diag_data.get('diagnosis_source'),
                        diag_data.get('diagnosis_context'),
                        diag_data.get('confidence_score'),
                        datetime.now()
                    ))
                    print(f"✓ Inserted diagnosis: {diag_data['diagnosis_name']}")
            
            # Insert Medications
            if data.get('medication'):
                for med_data in data['medication']:
                    med_query = """
                    INSERT INTO medications (patient_id, visit_id, medication_name, generic_name, dose, dose_unit,
                    frequency, route, start_date, end_date, is_active, is_prn, sig_text, patient_instructions, created_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(med_query, (
                        med_data.get('patient_id', patient_id),
                        primary_visit_id,
                        med_data['medication_name'],
                        med_data.get('generic_name'),
                        med_data.get('dose'),
                        med_data.get('dose_unit'),
                        med_data.get('frequency'),
                        med_data.get('route'),
                        self.parse_date(med_data.get('start_date')),
                        self.parse_date(med_data.get('end_date')),
                        med_data.get('is_active', True),
                        med_data.get('is_prn', False),
                        med_data.get('sig_text'),
                        med_data.get('patient_instructions'),
                        datetime.now()
                    ))
                    print(f"✓ Inserted medication: {med_data['medication_name']}")
            
            # Insert Symptoms
            if data.get('symptom'):
                for symptom_data in data['symptom']:
                    symptom_query = """
                    INSERT INTO symptoms (patient_id, visit_id, symptom_name, onset_date, duration, frequency,
                    severity, symptom_description, alleviating_factors, aggravating_factors, reported_date, resolution_date, created_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(symptom_query, (
                        symptom_data.get('patient_id', patient_id),
                        primary_visit_id,
                        symptom_data['symptom_name'],
                        self.parse_date(symptom_data.get('onset_date')),
                        symptom_data.get('duration'),
                        symptom_data.get('frequency'),
                        symptom_data.get('severity'),
                        symptom_data.get('symptom_description'),
                        symptom_data.get('alleviating_factors'),
                        symptom_data.get('aggravating_factors'),
                        self.parse_date(symptom_data.get('reported_date')),
                        self.parse_date(symptom_data.get('resolution_date')),
                        datetime.now()
                    ))
                    print(f"✓ Inserted symptom: {symptom_data['symptom_name']}")
            
            # Insert Vital Signs
            if data.get('vital_signs'):
                for vs_data in data['vital_signs']:
                    vs_query = """
                    INSERT INTO vital_signs (patient_id, visit_id, measurement_datetime, weight_kg, height_cm, bmi,
                    pulse_bpm, blood_pressure_systolic, blood_pressure_diastolic, temperature_celsius, 
                    respiratory_rate, oxygen_saturation_percent, pain_scale, created_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        primary_visit_id, 
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
                        vs_data.get('pain_scale'),
                        datetime.now()
                    ))
                    print(f"✓ Inserted vital signs for {measurement_dt}")
            
            # Insert Lab Results
            if data.get('lab_results'):
                for lab_data in data['lab_results']:
                    lab_query = """
                    INSERT INTO lab_results (patient_id, visit_id, test_name, test_value, reference_range,
                    units, test_date, result_status, created_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(lab_query, (
                        lab_data.get('patient_id', patient_id),
                        primary_visit_id,
                        lab_data.get('test_name'),
                        lab_data.get('test_value'),
                        lab_data.get('reference_range'),
                        lab_data.get('units'),
                        self.parse_date(lab_data.get('test_date')),
                        lab_data.get('result_status'),
                        datetime.now()
                    ))
                    print(f"✓ Inserted lab result: {lab_data.get('test_name')}")
            
            # Insert Imaging Studies
            if data.get('imaging_studies'):
                for img_data in data['imaging_studies']:
                    img_query = """
                    INSERT INTO imaging_studies (patient_id, visit_id, study_type, study_date, findings,
                    impression, created_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(img_query, (
                        img_data.get('patient_id', patient_id),
                        primary_visit_id,
                        img_data.get('study_type'),
                        self.parse_date(img_data.get('study_date')),
                        img_data.get('findings'),
                        img_data.get('impression'),
                        datetime.now()
                    ))
                    print(f"✓ Inserted imaging study: {img_data.get('study_type')}")
            
            # Insert Procedures
            if data.get('procedures'):
                for proc_data in data['procedures']:
                    proc_query = """
                    INSERT INTO procedures (patient_id, visit_id, procedure_name, procedure_date, provider_name,
                    procedure_notes, created_date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(proc_query, (
                        proc_data.get('patient_id', patient_id),
                        primary_visit_id,
                        proc_data.get('procedure_name'),
                        self.parse_date(proc_data.get('procedure_date')),
                        proc_data.get('provider_name'),
                        proc_data.get('procedure_notes'),
                        datetime.now()
                    ))
                    print(f"✓ Inserted procedure: {proc_data.get('procedure_name')}")
            
            connection.commit()
            print(f"✓ Data successfully saved to MySQL database")
            print(f"✓ Primary visit ID used: {primary_visit_id}")
            
        except Exception as e:
            print(f"Error saving to MySQL: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and connection.is_connected():
                if cursor:
                    cursor.close()
                connection.close()