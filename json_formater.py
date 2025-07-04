import json
from copy import deepcopy

# ---------------------------------------
# Step 1: Extract nested Provider/Department
# ---------------------------------------
class JSONReviewer:
    def extract_linked_objects(self, input_json):
        provider_set = {}
        department_set = {}
        output_json = deepcopy(input_json)

        def recurse(self, obj):
            if isinstance(obj, dict):
                new_obj = {}
                for key, value in obj.items():
                    if isinstance(value, dict):
                        if "npi_number" in value and "provider_name" in value:
                            npi = value["npi_number"]
                            provider_set[npi] = value
                            new_obj[key] = {"provider_id": f"__provider_id__:{npi}"}
                        elif "department_name" in value:
                            dept_name = value["department_name"]
                            department_set[dept_name] = value
                            new_obj[key] = {"department_id": f"__department_id__:{dept_name}"}
                        else:
                            new_obj[key] = recurse(value)
                    elif isinstance(value, list):
                        new_obj[key] = [recurse(item) for item in value]
                    else:
                        new_obj[key] = value
                return new_obj
            elif isinstance(obj, list):
                return [recurse(item) for item in obj]
            else:
                return obj

        transformed = recurse(output_json)
        return transformed, {"providers": provider_set, "departments": department_set}

    # ---------------------------------------
    # Step 2: Resolve or generate IDs
    # ---------------------------------------
    def resolve_or_generate_ids(self, entities, existing_providers=None, existing_departments=None, id_start_provider=1000, id_start_dept=2000):
        id_map = {"providers": {}, "departments": {}}
        provider_counter = id_start_provider
        department_counter = id_start_dept

        existing_providers = existing_providers or {}
        existing_departments = existing_departments or {}

        for npi, data in entities["providers"].items():
            if npi in existing_providers:
                id_map["providers"][npi] = existing_providers[npi]
            else:
                id_map["providers"][npi] = provider_counter
                provider_counter += 1

        for name, data in entities["departments"].items():
            if name in existing_departments:
                id_map["departments"][name] = existing_departments[name]
            else:
                id_map["departments"][name] = department_counter
                department_counter += 1

        return id_map

    # ---------------------------------------
    # Step 3: Replace placeholders with IDs
    # ---------------------------------------
    def replace_placeholders_with_ids(self, json_obj, id_map):
        def recurse(obj):
            if isinstance(obj, dict):
                new_obj = {}
                for key, value in obj.items():
                    if isinstance(value, dict):
                        if "provider_id" in value and isinstance(value["provider_id"], str) and value["provider_id"].startswith("__provider_id__:"):
                            npi = value["provider_id"].split(":")[1]
                            new_obj[key] = {"provider_id": id_map["providers"].get(npi)}
                        elif "department_id" in value and isinstance(value["department_id"], str) and value["department_id"].startswith("__department_id__:"):
                            name = value["department_id"].split(":")[1]
                            new_obj[key] = {"department_id": id_map["departments"].get(name)}
                        else:
                            new_obj[key] = recurse(value)
                    elif isinstance(value, list):
                        new_obj[key] = [recurse(item) for item in value]
                    else:
                        new_obj[key] = value
                return new_obj
            elif isinstance(obj, list):
                return [recurse(item) for item in obj]
            else:
                return obj

        return recurse(json_obj)

# ---------------------------------------
# Example Usage
# ---------------------------------------
if __name__ == "__main__":
    # Sample input JSON
    full_data = {
        "visits": [
            {
                "visit_id": 1,
                "visit_date": "2024-01-01",
                "department_name": "Cardiology",
                "primary_provider": {
                    "provider_name": "Dr. Jane Smith",
                    "npi_number": "1234567890",
                    "specialty": "Cardiology",
                    "active_status": True
                }
            }
        ],
        "diagnoses": [
            {
                "diagnosis_name": "Hypertension",
                "diagnosing_provider": {
                    "provider_name": "Dr. Jane Smith",
                    "npi_number": "1234567890",
                    "specialty": "Cardiology",
                    "active_status": True
                }
            }
        ]
    }


#new object creation
    obj1= JSONReviewer()

    # Step 1: Extract and replace with placeholders
    transformed_json, refs = obj1.extract_linked_objects(full_data)

    # Step 2: Resolve or generate IDs
    id_mapping = obj1.resolve_or_generate_ids(
        refs,
        existing_providers={"1234567890": 9001},
        existing_departments={"Cardiology": 8001}
    )

    # Step 3: Finalize JSON with IDs
    final_json = obj1.replace_placeholders_with_ids(transformed_json, id_mapping)

    # Output result
    print(json.dumps(final_json, indent=2))
