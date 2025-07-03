from json_schemas import *
from pydantic import BaseModel
from typing import get_args, get_origin, Union

from langchain.prompts import PromptTemplate
import inspect
import json

class PromptGen:
    def get_prompt_template(self, model: type[BaseModel], patient_id: int) -> dict:
        def resolve_type(field_type):
            """Get base type, resolving Optional, Union, etc."""
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                non_none = [arg for arg in args if arg is not type(None)]
                return resolve_type(non_none[0]) if non_none else str
            return field_type

        def get_type_string(py_type):
            """Convert a Python type into a string."""
            if py_type is str:
                return "str"
            elif py_type is bool:
                return "bool"
            elif py_type is int:
                return "int"
            elif py_type is float:
                return "float"
            elif py_type is list:
                return "list"
            elif inspect.isclass(py_type) and issubclass(py_type, BaseModel):
                return "object"
            elif py_type is datetime:
                return "datetime (YYYY-MM-DDTHH:MM:SS)"
            else:
                return "unknown"

        def build_fields(model_cls, given_patient_id=patient_id):
            result = {}
            for name, field in model_cls.model_fields.items():
                # Skip other *_id fields except patient_id and visit_id
                if name.endswith("_id") and name not in ("patient_id", "visit_id"):
                    continue

                # Insert patient_id directly
                if name == "patient_id":
                    result[name] = given_patient_id
                    continue

                resolved_type = resolve_type(field.annotation)
                is_optional = get_origin(field.annotation) is Union and type(None) in get_args(field.annotation)

                if get_origin(resolved_type) is list:
                    inner_type = get_args(resolved_type)[0]
                    entry = {
                        "type": "list",
                        "items": build_fields(inner_type, given_patient_id) if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel)
                        else { "type": get_type_string(inner_type), "optional": False }
                    }
                elif inspect.isclass(resolved_type) and issubclass(resolved_type, BaseModel):
                    entry = build_fields(resolved_type, given_patient_id)
                else:
                    entry = {
                        "type": get_type_string(resolved_type),
                        "optional": is_optional
                    }

                result[name] = entry
            return result

        return build_fields(model)

    def generate_json_prompt (self, schemas, patient_id):
        json_prompt_template = {
            schema.__name__.lower(): self.get_prompt_template(schema, patient_id)
            for schema in schemas
        }
        return json_prompt_template


if __name__ == "__main__":
    print("hello")