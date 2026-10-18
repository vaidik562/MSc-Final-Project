
import re, sqlite3, hashlib, hmac, secrets
from datetime import datetime
from modules.database_module import connect

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def validate_password(password, confirm=None):
    errors=[]
    if len(password)<8: errors.append("Password must contain at least 8 characters.")
    if not any(c.isalpha() for c in password): errors.append("Password must contain at least one letter.")
    if not any(c.isdigit() for c in password): errors.append("Password must contain at least one number.")
    if confirm is not None and password != confirm: errors.append("Passwords do not match.")
    return errors

def validate_registration(name,email,password,confirm):
    errors=[]
    if len(name.strip())<2: errors.append("Please enter a valid full name.")
    if not EMAIL_RE.match(email.strip()): errors.append("Please enter a valid email address.")
    errors.extend(validate_password(password, confirm))
    return errors

def _hash(password,salt):
    return hashlib.pbkdf2_hmac("sha256",password.encode("utf-8"),bytes.fromhex(salt),180000).hex()

def register_user(name,email,password):
    salt=secrets.token_hex(16)
    pw=_hash(password,salt)
    con=connect()
    try:
        con.execute(
            "INSERT INTO users(full_name,email,password_hash,salt,created_at) VALUES(?,?,?,?,?)",
            (name.strip(),email.strip().lower(),pw,salt,datetime.now().isoformat(timespec="seconds"))
        )
        con.commit()
        return True,"Registration successful. Please login."
    except sqlite3.IntegrityError:
        return False,"An account with this email already exists."
    finally:
        con.close()

def login_user(email,password):
    con=connect()
    row=con.execute(
        "SELECT id,full_name,email,password_hash,salt FROM users WHERE email=?",
        (email.strip().lower(),)
    ).fetchone()
    con.close()
    if not row: return None
    uid,name,mail,stored,salt=row
    if hmac.compare_digest(_hash(password,salt),stored):
        return {"id":uid,"full_name":name,"email":mail}
    return None


def reset_user_password(email, new_password):
    """Reset a local account password after the caller has validated the new password.

    This project is designed for local/academic use, so account ownership is checked only
    by the registered email address. A production deployment should replace this flow with
    an emailed one-time token/OTP before allowing the update.
    """
    normalized_email = email.strip().lower()
    if not EMAIL_RE.match(normalized_email):
        return False, "Please enter a valid email address."

    salt = secrets.token_hex(16)
    password_hash = _hash(new_password, salt)
    con = connect()
    try:
        cur = con.execute(
            "UPDATE users SET password_hash=?, salt=? WHERE email=?",
            (password_hash, salt, normalized_email),
        )
        con.commit()
        if cur.rowcount == 0:
            return False, "No account was found with this email address."
        return True, "Password reset successful. You can now login with your new password."
    finally:
        con.close()
