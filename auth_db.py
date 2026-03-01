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
                is_active BOOLEAN DEFAULT 1,
                is_super_admin BOOLEAN DEFAULT 0,
                trial_ends_at TIMESTAMP
            )
        ''')
        # Ensure email is unique (create index) - handle existing duplicates gracefully
        try:
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL
            ''')
        except sqlite3.IntegrityError:
            conn.commit()
            conn.close()
            # Normalize duplicates then retry
            self.normalize_duplicate_emails()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL
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
        
        # Activity logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feature usage aggregate
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                username TEXT,
                use_count INTEGER DEFAULT 0,
                last_used TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_usage ON feature_usage(feature_name, username)
        ''')
        
        # App settings (SMTP configuration)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS app_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                smtp_host TEXT,
                smtp_port INTEGER,
                smtp_user TEXT,
                smtp_pass TEXT,
                smtp_sender TEXT,
                smtp_tls INTEGER DEFAULT 1
            )
        ''')
        
        # Create OTP table for email verification
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS otp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                otp_code TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        conn.commit()
        # Migrate schema to ensure required columns exist on existing databases
        try:
            self.migrate_schema()
        except Exception:
            pass
        conn.close()
        
        # Create default super admin if missing
        self.ensure_super_admin_exists()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def check_password_strength(self, password):
        """
        Check password strength and return score (0-4) and feedback message.
        """
        if not password:
            return 0, "Password kosong"
            
        score = 0
        feedback = []
        
        # Length check
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("Minimal 8 karakter")
            
        # Uppercase check
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Gunakan huruf kapital (A-Z)")
            
        # Lowercase check
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Gunakan huruf kecil (a-z)")
            
        # Digit and Special char check
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in string.punctuation for c in password)
        
        if has_digit and has_special:
            score += 1
        elif has_digit:
            feedback.append("Tambahkan simbol/karakter khusus")
            score += 0.5
        elif has_special:
            feedback.append("Tambahkan angka (0-9)")
            score += 0.5
        else:
            feedback.append("Gunakan angka dan simbol")

        # Map score to labels
        if score < 2:
            label = "Sangat Lemah" if score == 0 else "Lemah"
        elif score < 3:
            label = "Sedang"
        elif score < 4:
            label = "Kuat"
        else:
            label = "Sangat Kuat"
            
        return score, label, feedback

    def generate_strong_password(self, length=14):
        """Generate a random strong password"""
        alphabet = string.ascii_letters + string.digits + string.punctuation
        # Ensure at least one of each required type
        password = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
            secrets.choice(string.punctuation)
        ]
        # Fill the rest
        password += [secrets.choice(alphabet) for _ in range(length - 4)]
        # Shuffle to randomize positions
        secrets.SystemRandom().shuffle(password)
        return "".join(password)

    def create_user(self, username, password, email=None):
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prevent duplicate email proactively
            if email:
                cursor.execute('SELECT COUNT(*) FROM users WHERE email = ?', (email,))
                if cursor.fetchone()[0] > 0:
                    conn.close()
                    return False
            
            password_hash = self.hash_password(password)
            trial_ends_at = (datetime.now() + timedelta(days=30)).isoformat()
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, trial_ends_at)
                VALUES (?, ?, ?, ?)
            ''', (username, password_hash, email, trial_ends_at))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_user_by_username(self, username):
        """Fetch user record by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, is_active, is_super_admin, trial_ends_at FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                'id': row[0], 
                'username': row[1], 
                'email': row[2], 
                'is_active': row[3], 
                'is_super_admin': row[4],
                'trial_ends_at': row[5]
            }
        return None
    
    def normalize_duplicate_emails(self):
        """Find duplicate emails and set to NULL for non-primary records to satisfy unique index"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Find duplicate emails
        cursor.execute('''
            SELECT email FROM users 
            WHERE email IS NOT NULL
            GROUP BY email
            HAVING COUNT(*) > 1
        ''')
        duplicates = [row[0] for row in cursor.fetchall()]
        for email in duplicates:
            # Keep the earliest created (lowest id) and nullify others
            cursor.execute('SELECT id FROM users WHERE email = ? ORDER BY id ASC', (email,))
            ids = [r[0] for r in cursor.fetchall()]
            # Nullify all except first
            for user_id in ids[1:]:
                cursor.execute('UPDATE users SET email = NULL WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
    
    def migrate_schema(self):
        """Ensure required columns exist in legacy databases."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Check users table columns
        cursor.execute("PRAGMA table_info(users)")
        cols = [row[1] for row in cursor.fetchall()]
        # Add is_super_admin if missing
        if 'is_super_admin' not in cols:
            cursor.execute("ALTER TABLE users ADD COLUMN is_super_admin INTEGER DEFAULT 0")
            cursor.execute("UPDATE users SET is_super_admin = 0 WHERE is_super_admin IS NULL")
        
        # Add trial_ends_at if missing
        if 'trial_ends_at' not in cols:
            cursor.execute("ALTER TABLE users ADD COLUMN trial_ends_at TIMESTAMP")
            # Set default trial for existing users (30 days from now, as a courtesy)
            trial_end = (datetime.now() + timedelta(days=30)).isoformat()
            cursor.execute("UPDATE users SET trial_ends_at = ? WHERE trial_ends_at IS NULL", (trial_end,))
        # Ensure indexes (partial unique for non-NULL emails)
        try:
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL
            ''')
        except sqlite3.OperationalError:
            # Fallback: attempt normal unique index after normalization
            self.normalize_duplicate_emails()
            try:
                cursor.execute('''
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)
                ''')
            except Exception:
                pass
        conn.commit()
        conn.close()
    
    def get_user_by_email(self, email):
        """Fetch user record by email"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, is_active, is_super_admin FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {'id': row[0], 'username': row[1], 'email': row[2], 'is_active': row[3], 'is_super_admin': row[4]}
        return None
    
    def is_super_admin(self, username) -> bool:
        """Check if user is super admin"""
        u = self.get_user_by_username(username)
        return bool(u and u.get('is_super_admin'))
    
    def ensure_super_admin_exists(self):
        """Ensure there is at least one super admin account, create default if none"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users WHERE is_super_admin = 1')
            count = cursor.fetchone()[0]
            if count == 0:
                # Get from environment or defaults
                username = os.getenv('SUPER_ADMIN_USER', 'superadmin')
                password = os.getenv('SUPER_ADMIN_PASS', 'Admin@12345')
                email = os.getenv('SUPER_ADMIN_EMAIL', 'admin@local')
                pw_hash = self.hash_password(password)
                cursor.execute('''
                    INSERT OR IGNORE INTO users (username, password_hash, email, is_active, is_super_admin)
                    VALUES (?, ?, ?, 1, 1)
                ''', (username, pw_hash, email))
                conn.commit()
            conn.close()
        except Exception:
            pass
    
    def authenticate_user(self, username, password):
        """Authenticate user and return user info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, password_hash, failed_attempts, locked_until, is_active, trial_ends_at
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return None
        
        user_id, username, stored_hash, failed_attempts, locked_until, is_active, trial_ends_at = user
        
        # Check if account is locked
        if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
            return {'error': 'locked', 'locked_until': locked_until}
        
        # Check if account is active
        if not is_active:
            return None
        
        # Verify password
        if self.hash_password(password) == stored_hash:
            # Reset failed attempts on successful login
            self.reset_failed_attempts(username)
            return {
                'id': user_id, 
                'username': username,
                'trial_ends_at': trial_ends_at
            }
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
                    WHEN failed_attempts >= 5 THEN datetime('now', '+15 minutes')
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
    
    def generate_otp(self, username, length: int = 6, ttl_minutes: int = 10):
        """Generate and store OTP code for the given user, returns (code, expires_at)"""
        user = self.get_user_by_username(username)
        if not user or not user.get('email'):
            return None
        
        # Create OTP code (digits only)
        digits = string.digits
        otp_code = ''.join(secrets.choice(digits) for _ in range(length))
        expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Remove previous OTPs for this user
        cursor.execute('DELETE FROM otp_codes WHERE username = ?', (username,))
        # Insert new OTP
        cursor.execute('''
            INSERT INTO otp_codes (username, email, otp_code, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (username, user['email'], otp_code, expires_at))
        conn.commit()
        conn.close()
        
        return {'code': otp_code, 'email': user['email'], 'expires_at': expires_at}
    
    def verify_otp(self, username, code):
        """Verify OTP code for user; returns True on success, False otherwise"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT otp_code, expires_at FROM otp_codes
            WHERE username = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (username,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False
        otp_code, expires_at = row
        # Check expiration
        try:
            exp_dt = datetime.fromisoformat(expires_at)
        except Exception:
            # SQLite may return already datetime
            exp_dt = expires_at if isinstance(expires_at, datetime) else datetime.now() - timedelta(days=1)
        is_valid = (str(code).strip() == str(otp_code).strip()) and (exp_dt > datetime.now())
        if is_valid:
            # consume OTP
            cursor.execute('DELETE FROM otp_codes WHERE username = ?', (username,))
            conn.commit()
        conn.close()
        return is_valid
    
    def record_activity(self, username, action: str, metadata: str = None):
        """Record an activity event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO activity_logs (username, action, metadata)
            VALUES (?, ?, ?)
        ''', (username, action, metadata))
        conn.commit()
        conn.close()
    
    def record_feature_usage(self, username, feature_name: str):
        """Increment feature usage count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Upsert-like behavior
        cursor.execute('''
            SELECT id, use_count FROM feature_usage WHERE feature_name = ? AND username = ?
        ''', (feature_name, username))
        row = cursor.fetchone()
        if row:
            new_count = (row[1] or 0) + 1
            cursor.execute('''
                UPDATE feature_usage SET use_count = ?, last_used = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_count, row[0]))
        else:
            cursor.execute('''
                INSERT INTO feature_usage (feature_name, username, use_count, last_used)
                VALUES (?, ?, 1, CURRENT_TIMESTAMP)
            ''', (feature_name, username))
        conn.commit()
        conn.close()
    
    def get_users_dataframe(self):
        """Return users as list of dicts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, is_active, is_super_admin, created_at, last_login FROM users')
        rows = cursor.fetchall()
        conn.close()
        cols = ['id','username','email','is_active','is_super_admin','created_at','last_login']
        return [dict(zip(cols, r)) for r in rows]
    
    def get_feature_usage_stats(self):
        """Return feature usage stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT feature_name, SUM(use_count) as total, MAX(last_used) FROM feature_usage GROUP BY feature_name ORDER BY total DESC')
        rows = cursor.fetchall()
        conn.close()
        return [{'feature_name': r[0], 'total': r[1], 'last_used': r[2]} for r in rows]
    
    def get_activity_summary(self):
        """Return recent activity summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT action, COUNT(*) as total FROM activity_logs GROUP BY action ORDER BY total DESC')
        rows = cursor.fetchall()
        conn.close()
        return [{'action': r[0], 'total': r[1]} for r in rows]
    
    # Admin operations
    def change_password(self, username, old_password, new_password):
        """Change user's password after verifying old password"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False
        current_hash = row[0]
        if self.hash_password(old_password) != current_hash:
            conn.close()
            return False
        new_hash = self.hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = ? WHERE username = ?', (new_hash, username))
        conn.commit()
        conn.close()
        return True
    
    def delete_user(self, username):
        """Delete user and related sessions/OTP"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE username = ?', (username,))
        cursor.execute('DELETE FROM otp_codes WHERE username = ?', (username,))
        cursor.execute('DELETE FROM users WHERE username = ?', (username,))
        conn.commit()
        conn.close()
        return True
    
    def set_user_active(self, username, is_active: bool):
        """Activate or deactivate user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_active = ? WHERE username = ?', (1 if is_active else 0, username))
        conn.commit()
        conn.close()
        return True
    
    def set_user_super_admin(self, username, is_super_admin: bool):
        """Grant or revoke super admin role"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_super_admin = ? WHERE username = ?', (1 if is_super_admin else 0, username))
        conn.commit()
        conn.close()
        return True
    
    # SMTP settings
    def get_smtp_config(self):
        """Get SMTP configuration from app settings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT smtp_host, smtp_port, smtp_user, smtp_pass, smtp_sender, smtp_tls FROM app_settings WHERE id = 1')
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {}
        return {
            'host': row[0],
            'port': row[1],
            'user': row[2],
            'password': row[3],
            'sender': row[4],
            'tls': bool(row[5]) if row[5] is not None else True
        }
    
    def set_smtp_config(self, cfg: dict):
        """Upsert SMTP configuration into app settings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM app_settings WHERE id = 1')
        exists = cursor.fetchone()[0] > 0
        if exists:
            cursor.execute('''
                UPDATE app_settings 
                SET smtp_host = ?, smtp_port = ?, smtp_user = ?, smtp_pass = ?, smtp_sender = ?, smtp_tls = ?
                WHERE id = 1
            ''', (cfg.get('host'), cfg.get('port'), cfg.get('user'), cfg.get('password'), cfg.get('sender'), 1 if cfg.get('tls', True) else 0))
        else:
            cursor.execute('''
                INSERT INTO app_settings (id, smtp_host, smtp_port, smtp_user, smtp_pass, smtp_sender, smtp_tls)
                VALUES (1, ?, ?, ?, ?, ?, ?)
            ''', (cfg.get('host'), cfg.get('port'), cfg.get('user'), cfg.get('password'), cfg.get('sender'), 1 if cfg.get('tls', True) else 0))
        conn.commit()
        conn.close()
        return True
    
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
auth_db = AuthDatabase()
