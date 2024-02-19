
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

from config import databaseURL

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': databaseURL
})

ref = db.reference('Students')

data = {
    "2018267":
        {
            "name": "Ayush Negi",
            "major": "Computer",
            "starting_year": 2020,
            "total_attendance": 7,
            "standing": "A",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "852741":
        {
            "name": "Piyush Bidaliya",
            "major": "Economics",
            "starting_year": 2021,
            "total_attendance": 12,
            "standing": "B",
            "year": 1,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "963852":
        {
            "name": "Aman Bisht",
            "major": "Physics",
            "starting_year": 2020,
            "total_attendance": 7,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2022-12-11 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)

