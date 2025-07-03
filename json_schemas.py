from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Patient(BaseModel):
    patient_id:             int                     = Field(..., description="Primary key")
    medical_record_number:  Optional[str]           = Field(None, description="Unique medical record number")
    created_date:           datetime                = Field(default_factory=datetime.now)
    updated_date:           Optional[datetime]      = Field(None)

class Provider(BaseModel):
    provider_name:          Optional[str]           = Field(None, description="Name of provider offering care")
    npi_number:             Optional[str]           = Field(None, description="Unique NPI number")
    specialty:              Optional[str]           = Field(None)
    department_id:          Optional[int]           = Field(None)
    active_status:          bool                    = Field(default=True)
    created_date:           datetime                = Field(default_factory=datetime.now)

class Department(BaseModel):
    department_name:        Optional[str]           = Field(None)
    department_type:        Optional[str]           = Field(None)
    system_name:            Optional[str]           = Field(None)
    created_date:           datetime                = Field(default_factory=datetime.now)

class Visit(BaseModel):
    visit_id:               int                     = Field(..., description="Primary key")
    patient_id:             Optional[int]           = Field(None)
    visit_date:             Optional[str]           = Field(None)
    visit_type:             Optional[str]           = Field(None)
    visit_notes:            Optional[VisitNotes]    = Field(None)
    department:             Optional[Department]    = None
    primary_provider:       Optional[Provider]      = None
    discharge_date:         Optional[datetime]      = Field(None)
    created_date:           datetime                = Field(default_factory=datetime.now)

class VisitNotes(BaseModel):
    patient_id:             Optional[int]           = Field(None)
    note_date:              Optional[str]           = Field(None)
    note_type:              Optional[str]           = Field(None)
    full_note_text:         Optional[str]           = Field(None)
    chief_complaint:        Optional[str]           = Field(None)
    history_present_illness:Optional[str]           = Field(None)
    review_of_systems:      Optional[str]           = Field(None)
    physical_exam:          Optional[str]           = Field(None)
    assessment:             Optional[str]           = Field(None)
    plan:                   Optional[str]           = Field(None)
    author_provider:        Optional[Provider]      = None
    extraction_confidence:  Optional[float]         = Field(None, ge=0.0, le=1.0)
    extraction_method:      Optional[str]           = Field(None)
    extraction_timestamp:   Optional[str]           = Field(None)
    created_date:           datetime                = Field(default_factory=datetime.now)

class Diagnosis(BaseModel):
    patient_id:             Optional[int]           = Field(None)
    visit_id:               Optional[int]           = Field(None)
    diagnosis_name:         str
    icd10_code:             Optional[str]           = Field(None)
    onset_date:             Optional[str]           = Field(None)
    resolution_date:        Optional[str]           = Field(None)
    is_chronic:             Optional[bool]          = Field(default=False)
    is_active:              Optional[bool]          = Field(default=True)
    severity:               Optional[str]           = Field(None)
    diagnosing_provider:    Optional[Provider]      = None
    diagnosis_source:       Optional[str]           = Field(None)
    diagnosis_context:      Optional[str]           = Field(None)
    confidence_score:       Optional[float]         = Field(None, ge=0.0, le=1.0)
    updated_date:           Optional[str]           = Field(None)
    created_date:           datetime                = Field(default_factory=datetime.now)

class Symptom(BaseModel):
    patient_id:             Optional[int]           = Field(None)
    visit_id:               Optional[int]           = Field(None)
    symptom_name:           str
    onset_date:             Optional[str]           = Field(None)
    duration:               Optional[str]           = Field(None)
    frequency:              Optional[str]           = Field(None)
    severity:               Optional[str]           = Field(None)
    symptom_description:    Optional[str]           = Field(None)
    alleviating_factors:    Optional[str]           = Field(None)
    aggravating_factors:    Optional[str]           = Field(None)
    reported_date:          Optional[str]           = Field(None)
    resolution_date:        Optional[str]           = Field(None)
    created_date:           datetime                = Field(default_factory=datetime.now)

class Medication(BaseModel):
    patient_id:             Optional[int]           = Field(None)
    visit_id:               Optional[int]           = Field(None)
    medication_name:        str
    generic_name:           Optional[str]           = Field(None)
    rxnorm_code:            Optional[str]           = Field(None)
    dose:                   Optional[str]           = Field(None)
    dose_unit:              Optional[str]           = Field(None)
    frequency:              Optional[str]           = Field(None)
    route:                  Optional[str]           = Field(None)
    start_date:             Optional[str]           = Field(None)
    end_date:               Optional[str]           = Field(None)
    discontinuation_reason: Optional[str]           = Field(None)
    is_active:              Optional[bool]          = Field(default=True)
    is_prn:                 Optional[bool]          = Field(default=False)
    prescribing_provider:   Optional[Provider]      = None
    sig_text:               Optional[str]           = Field(None)
    patient_instructions:   Optional[str]           = Field(None)
    updated_date:           Optional[str]           = Field(None)
    created_date:           datetime                = Field(default_factory=datetime.now)

class VitalSigns(BaseModel):
    patient_id:             Optional[int]           = Field(None)
    visit_id:               Optional[int]           = Field(None)
    measurement_datetime:   str
    weight_kg:              Optional[float]         = Field(None, ge=0)
    height_cm:              Optional[float]         = Field(None, ge=0)
    bmi:                    Optional[float]         = Field(None, ge=0)
    pulse_bpm:              Optional[int]           = Field(None, ge=0)
    blood_pressure_systolic:Optional[int]           = Field(None, ge=0)
    blood_pressure_diastolic:Optional[int]          = Field(None, ge=0)
    temperature_celsius:    Optional[float]         = Field(None, ge=0)
    respiratory_rate:       Optional[int]           = Field(None, ge=0)
    oxygen_saturation_percent:Optional[int]         = Field(None, ge=0, le=100)
    pain_scale:             Optional[int]           = Field(None, ge=0, le=10)
    additional_vitals:      Optional[str]           = Field(None)
    measurement_context:    Optional[str]           = Field(None)
    measured_by:            Optional[Provider]      = None
    created_date:           datetime                = Field(default_factory=datetime.now)

class LabResult(BaseModel):
    patient_id:             Optional[int]           = Field(None)
    visit_id:               Optional[int]           = Field(None)
    lab_name:               str
    test_name:              str
    loinc_code:             Optional[str]           = Field(None)
    result_value:           str
    result_numeric:         Optional[float]         = Field(None)
    unit_of_measurement:    Optional[str]           = Field(None)
    reference_range_low:    Optional[float]         = Field(None)
    reference_range_high:   Optional[float]         = Field(None)
    reference_range_text:   Optional[str]           = Field(None)
    abnormality_flag:       Optional[bool]          = Field(default=False)
    abnormality_type:       Optional[str]           = Field(None)
    collection_datetime:    Optional[str]           = Field(None)
    result_datetime:        Optional[str]           = Field(None)
    ordering_provider:      Optional[Provider]      = None
    result_status:          Optional[str]           = Field(None)
    clinical_significance:  Optional[str]           = Field(None)
    created_date:           datetime                = Field(default_factory=datetime.now)

class ImagingStudy(BaseModel):
    patient_id:             Optional[int]           = Field(None)
    visit_id:               Optional[int]           = Field(None)
    imaging_type:           str
    modality:               Optional[str]           = Field(None)
    body_region:            Optional[str]           = Field(None)
    study_datetime:         str
    ordering_provider:      Optional[Provider]      = None
    radiologist:            Optional[Provider]      = None
    indication:             Optional[str]           = Field(None)
    technique:              Optional[str]           = Field(None)
    comparison:             Optional[str]           = Field(None)
    findings:               Optional[str]           = Field(None)
    impression:             Optional[str]           = Field(None)
    key_findings:           Optional[str]           = Field(None)
    report_status:          Optional[str]           = Field(None)
    critical_findings:      Optional[bool]          = Field(default=False)
    created_date:           datetime                = Field(default_factory=datetime.now)

class ProcedureTreatment(BaseModel):
    patient_id:             Optional[int]           = Field(None)
    visit_id:               Optional[int]           = Field(None)
    procedure_name:         str
    procedure_type:         Optional[str]           = Field(None)
    cpt_code:               Optional[str]           = Field(None)
    procedure_date:         str
    duration_minutes:       Optional[int]           = Field(None, ge=0)
    outcome:                Optional[str]           = Field(None)
    outcome_details:        Optional[str]           = Field(None)
    complications:          Optional[str]           = Field(None)
    primary_provider:       Optional[Provider]      = None
    therapy_type:           Optional[str]           = Field(None)
    sessions_completed:     Optional[int]           = Field(None, ge=0)
    sessions_planned:       Optional[int]           = Field(None, ge=0)
    created_date:           datetime                = Field(default_factory=datetime.now)
