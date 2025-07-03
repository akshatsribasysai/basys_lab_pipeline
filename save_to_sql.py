from json_schemas import *
import json
import mysql.connector

class SQLSaver:

    def isolate_jsons (self, json_str, patient_id):
    # Extract and validate JSON with Pydantic

        # try:
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
            
            # try:
            print(raw_data['visit'])
            if raw_data.get('visit'):
                validated_data['visit'] = [Visit(**raw_data['visit'])]
                print(f"✓ Validated {len(validated_data['visit'])} visit")

            if raw_data.get('visit_notes'):
                validated_data['visit_notes'] = [VisitNotes(**vn) for vn in raw_data['visit_notes']]
                print(f"✓ Validated {len(validated_data['visit_notes'])} visit notes")

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
            print("✓ All data validated successfully with Pydantic")
            validated_data = raw_data
            print(validated_data)
            return validated_data
                
        #     except ValidationError as ve:
        #         print(f"Pydantic validation error: {ve}")
        #         print("Raw data that failed validation:")
        #         print(json.dumps(raw_data, indent=2, default=str))
        #         # Return raw data even if validation fails, so we can see what was extracted
        #         return raw_data
            
        #     except json.JSONDecodeError as e:
        #         print(f"JSON parsing error: {e}")
        #         print(f"Raw response: {json_str}")
        #         return None
        
        # except:

        #     print("No valid JSON found in response")
        #     print(f"Raw response: {json_str}")
        #     return None


    def save_to_mysql(self, json_data, db_config, patient_id: int):
        """Save extracted JSON data to MySQL database"""
        connection = None
        cursor = None
        
        # try:
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
            
        # except:
        #     print(f"Error saving to MySQL")
        #     if connection:
        #         connection.rollback()
        # finally:
        #     if connection and connection.is_connected():
        #         if cursor:
        #             cursor.close()
        #         connection.close()

