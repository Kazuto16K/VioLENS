from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from . import db
import bcrypt
from flask_login import login_user, login_required, login_user, current_user, logout_user

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if bcrypt.checkpw(password.encode('utf-8'),user.password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.',category='alert')
        else:
            flash('User does not exist.',category='alert')

    return render_template("login.html", user=current_user)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('User logged out successfully.',category='success')
    return redirect(url_for('views.index'))

@auth.route('/sign-up', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        phone = request.form.get('phone')
        alert_phone = request.form.get('alert-phone')

        user = User.query.filter_by(email=email).first()
        if user:
            flash("Email already exists", category='alert')
        elif len(email) < 4:
            flash("Email must be greater than 4 characters", category='alert')
        elif len(name) < 2:
            flash("Name must be greater than 2 characters", category='alert')
        elif password1 != password2:
            flash("Passwords don't match", category='alert')
        elif len(password1) < 5:
            flash("Passwords must be atleast 6 characters", category='alert')
        elif len(phone) != 10:
            flash("Phone Number must be exactly 10 digits", category='alert')
        elif len(alert_phone) != 10:
            flash("Alert Phone Number must be exactly 10 digits", category='alert')
        else:
            new_user = User(email=email, 
                            name=name, 
                            password=bcrypt.hashpw(password1.encode('utf-8'), bcrypt.gensalt()), 
                            phone = phone, 
                            alert_phone_number=alert_phone)
            db.session.add(new_user)
            db.session.commit()

            flash('Account Created!', category='success')
            login_user(new_user, remember=False)
            return redirect(url_for('views.home'))

    return render_template("signup.html", user=current_user)

@auth.route('/dashboard')
@login_required
def dashboard():
    return render_template("dashboard.html",user=current_user)

@auth.route('/update_details', methods=['GET','POST'])
@login_required
def update_details():
    if request.method == 'POST':
        current_pass = current_user.password
        email = current_user.email
        phone = current_user.phone
        alert_phone = current_user.alert_phone_number

        new_phone = request.form.get('phone')
        new_alert_phone = request.form.get('alert-phone')
        new_pass1 = request.form.get('password1')
        new_pass2 = request.form.get('password2')
        alert_email = request.form.get('alert-email')

        check_pass = request.form.get('current-password')
        if current_user:
            if bcrypt.checkpw(check_pass.encode('utf-8'),current_pass):

                new_phone = phone if phone is None else new_phone              
                new_alert_phone = alert_phone if alert_phone is None else new_alert_phone

                if new_pass1 == new_pass2:
                    current_pass = new_pass1 if new_pass1 is not None else current_pass
                    hashed_password = bcrypt.hashpw(current_pass.encode('utf-8'), bcrypt.gensalt())
                    user = User.query.get(current_user.id)
                    user.password = hashed_password
                    user.phone = new_phone
                    user.alert_phone_number = new_alert_phone
                    db.session.commit()
                    flash("Details updated successfully!", category="success")
                else:
                    flash("New passwords don't match!", category="alert")
        
        return redirect(url_for('auth.dashboard'))
