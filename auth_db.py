<<<<<<< HEAD
import sqlite3
import hashlib
import secrets
import string
from datetime import datetime, timedelta
import os

class AuthDatabase:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with users table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create sessions table for tracking active sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, password, email=None):
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, email)
                VALUES (?, ?, ?)
            ''', (username, password_hash, email))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def authenticate_user(self, username, password):
        """Authenticate user and return user info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, password_hash, failed_attempts, locked_until, is_active
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return None
        
        user_id, username, stored_hash, failed_attempts, locked_until, is_active = user
        
        # Check if account is locked
        if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
            return None
        
        # Check if account is active
        if not is_active:
            return None
        
        # Verify password
        if self.hash_password(password) == stored_hash:
            # Reset failed attempts on successful login
            self.reset_failed_attempts(username)
            return {'id': user_id, 'username': username}
        else:
            # Increment failed attempts
            self.increment_failed_attempts(username)
            return None
    
    def increment_failed_attempts(self, username):
        """Increment failed login attempts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET failed_attempts = failed_attempts + 1,
                locked_until = CASE 
                    WHEN failed_attempts >= 4 THEN datetime('now', '+30 minutes')
                    ELSE locked_until
                END
            WHERE username = ?
        ''', (username,))
        
        conn.commit()
        conn.close()
    
    def reset_failed_attempts(self, username):
        """Reset failed attempts on successful login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET failed_attempts = 0,
                locked_until = NULL,
                last_login = CURRENT_TIMESTAMP
            WHERE username = ?
        ''', (username,))
        
        conn.commit()
        conn.close()
    
    def create_session(self, username):
        """Create a new session token"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (username, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (username, token, expires_at))
        
        conn.commit()
        conn.close()
        
        return token
    
    def validate_session(self, token):
        """Validate session token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, expires_at
            FROM sessions
            WHERE session_token = ? AND expires_at > datetime('now')
        ''', (token,))
        
        session = cursor.fetchone()
        conn.close()
        
        if session:
            return {'username': session[0]}
        return None
    
    def delete_session(self, token):
        """Delete session (logout)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM sessions WHERE session_token = ?', (token,))
        
        conn.commit()
        conn.close()
    
    def is_username_available(self, username):
        """Check if username is available"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count == 0

# Initialize global auth database
=======
import sqlite3
import hashlib
import secrets
import string
from datetime import datetime, timedelta
import os

class AuthDatabase:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with users table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create sessions table for tracking active sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, password, email=None):
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, email)
                VALUES (?, ?, ?)
            ''', (username, password_hash, email))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def authenticate_user(self, username, password):
        """Authenticate user and return user info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, password_hash, failed_attempts, locked_until, is_active
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return None
        
        user_id, username, stored_hash, failed_attempts, locked_until, is_active = user
        
        # Check if account is locked
        if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
            return None
        
        # Check if account is active
        if not is_active:
            return None
        
        # Verify password
        if self.hash_password(password) == stored_hash:
            # Reset failed attempts on successful login
            self.reset_failed_attempts(username)
            return {'id': user_id, 'username': username}
        else:
            # Increment failed attempts
            self.increment_failed_attempts(username)
            return None
    
    def increment_failed_attempts(self, username):
        """Increment failed login attempts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET failed_attempts = failed_attempts + 1,
                locked_until = CASE 
                    WHEN failed_attempts >= 4 THEN datetime('now', '+30 minutes')
                    ELSE locked_until
                END
            WHERE username = ?
        ''', (username,))
        
        conn.commit()
        conn.close()
    
    def reset_failed_attempts(self, username):
        """Reset failed attempts on successful login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET failed_attempts = 0,
                locked_until = NULL,
                last_login = CURRENT_TIMESTAMP
            WHERE username = ?
        ''', (username,))
        
        conn.commit()
        conn.close()
    
    def create_session(self, username):
        """Create a new session token"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (username, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (username, token, expires_at))
        
        conn.commit()
        conn.close()
        
        return token
    
    def validate_session(self, token):
        """Validate session token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, expires_at
            FROM sessions
            WHERE session_token = ? AND expires_at > datetime('now')
        ''', (token,))
        
        session = cursor.fetchone()
        conn.close()
        
        if session:
            return {'username': session[0]}
        return None
    
    def delete_session(self, token):
        """Delete session (logout)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM sessions WHERE session_token = ?', (token,))
        
        conn.commit()
        conn.close()
    
    def is_username_available(self, username):
        """Check if username is available"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count == 0

# Initialize global auth database
>>>>>>> 82a9a7837d554aa663a3debb5b0cd475375882e8
auth_db = AuthDatabase()