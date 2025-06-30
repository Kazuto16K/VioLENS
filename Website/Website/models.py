from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    image_path = db.Column(db.String(255))
    

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    email = db.Column(db.String(150), unique = True)
    password = db.Column(db.String(150))
    name = db.Column(db.String(150))
    phone = db.Column(db.String(10))
    alert_phone_number = db.Column(db.String(10))
    detections = db.relationship('Detection')

