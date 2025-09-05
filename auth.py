import os
import json
import bcrypt
import streamlit as st

USERS_FILE = os.path.join("data", "users.json")

def get_user_store():
    """Loads or creates the user data file."""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            # Create a default admin user with a hashed password
            hashed_password = bcrypt.hashpw(b"admin", bcrypt.gensalt()).decode('utf-8')
            default_user = {"admin": {"password": hashed_password, "role": "admin"}}
            json.dump(default_user, f, indent=4)
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_user_store(users):
    """Saves the user data to the file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def authenticate(username, password):
    """Checks if the provided credentials are valid."""
    users = get_user_store()
    if username in users:
        hashed_password = users[username]["password"].encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            return True
    return False

def register_user(username, password):
    """Registers a new user."""
    users = get_user_store()
    if username in users:
        return False
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    users[username] = {"password": hashed_password, "role": "user"}
    save_user_store(users)
    return True

def get_current_user():
    """Returns the username of the currently logged-in user."""
    return st.session_state.get('username')

def is_admin(username):
    """Checks if a user has admin privileges."""
    users = get_user_store()
    return users.get(username, {}).get("role") == "admin"