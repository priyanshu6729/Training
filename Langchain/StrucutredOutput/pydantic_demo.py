from pydantic import BaseModel , EmailStr, Field
from typing import List, Optional
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, description="CGPA must be between 0 and 10")

new_student = {
    "name": "John Doe",
    "age": 20,
    "email": "abc@example.com",
    "cgpa": 8.5 
}

student = Student(**new_student)
student_json = student.model_dump_json()
print(student_json)