import socket
import threading
import json
import pyttsx3
from datetime import datetime
import sqlite3
import uuid

# ============================================
# DATABASE MANAGER
# ============================================

class TransactionDatabase:
    """Manages SQLite database for transaction history"""
    
    def __init__(self, db_name='payment_system.db'):
        self.db_name = db_name
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                balance REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                sender TEXT NOT NULL,
                receiver TEXT NOT NULL,
                amount REAL NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL
            )
        ''')
        
        self.conn.commit()
    
    def create_or_update_user(self, username, balance, created_at, last_active):
        """Create new user or update existing user"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users (username, balance, created_at, last_active)
            VALUES (?, ?, ?, ?)
        ''', (username, balance, created_at, last_active))
        self.conn.commit()
    
    def get_user(self, username):
        """Retrieve user by username"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        return row if row else None
    
    def update_balance(self, username, new_balance):
        """Update user balance"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE users 
            SET balance = ?, last_active = ?
            WHERE username = ?
        ''', (new_balance, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), username))
        self.conn.commit()
    
    def add_transaction(self, transaction_id, sender, receiver, amount, timestamp, status):
        """Add new transaction to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO transactions 
            (transaction_id, sender, receiver, amount, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (transaction_id, sender, receiver, amount, timestamp, status))
        self.conn.commit()
    
    def get_user_transactions(self, username, limit=50):
        """Get all transactions for a user"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM transactions 
            WHERE sender = ? OR receiver = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (username, username, limit))
        return cursor.fetchall()
    
    def get_transaction_stats(self, username):
        """Get transaction statistics for user"""
        cursor = self.conn.cursor()
        
        # Total sent
        cursor.execute('''
            SELECT COALESCE(SUM(amount), 0) 
            FROM transactions 
            WHERE sender = ? AND status = 'completed'
        ''', (username,))
        total_sent = cursor.fetchone()[0]
        
        # Total received
        cursor.execute('''
            SELECT COALESCE(SUM(amount), 0) 
            FROM transactions 
            WHERE receiver = ? AND status = 'completed'
        ''', (username,))
        total_received = cursor.fetchone()[0]
        
        # Transaction count
        cursor.execute('''
            SELECT COUNT(*) 
            FROM transactions 
            WHERE (sender = ? OR receiver = ?) AND status = 'completed'
        ''', (username, username))
        transaction_count = cursor.fetchone()[0]
        
        return {
            'total_sent': total_sent,
            'total_received': total_received,
            'net_flow': total_received - total_sent,
            'transaction_count': transaction_count
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# ============================================
# PAYMENT SYSTEM
# ============================================

class PaymentSystem:
    def __init__(self, username, initial_balance=10000):
        self.username = username
        self.balance = initial_balance
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        self.socket = None
        self.connected = False
        self.peer_address = None
        self.peer_username = None
        self.db = TransactionDatabase()
        
        # Load or create user in database
        user = self.db.get_user(username)
        if user:
            self.balance = user[1]  # Load saved balance
            print(f"✓ Loaded existing user. Balance: ₹{self.balance:,.2f}")
        else:
            # Create new user
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.db.create_or_update_user(username, initial_balance, now, now)
            print(f"✓ Created new user with balance: ₹{initial_balance:,.2f}")
        
    def speak(self, text):
        """Text-to-speech announcement"""
        print(f"[ANNOUNCEMENT] {text}")
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except:
            pass  # Skip TTS if it fails
    
    def start_server(self, host='0.0.0.0', port=5555):
        """Start as server to receive connections"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        
        print(f"\n{'='*50}")
        print(f"Server started on {host}:{port}")
        print(f"Your IP: {socket.gethostbyname(socket.gethostname())}")
        print(f"Waiting for connection...")
        print(f"{'='*50}\n")
        
        self.socket, self.peer_address = server.accept()
        self.connected = True
        
        # Exchange usernames
        self.socket.send(json.dumps({'type': 'handshake', 'username': self.username}).encode('utf-8'))
        data = self.socket.recv(4096).decode('utf-8')
        message = json.loads(data)
        self.peer_username = message.get('username', 'Unknown')
        
        print(f"✓ Connected to {self.peer_username} at {self.peer_address}")
        self.speak(f"Connection established with {self.peer_username}")
        
        # Start listening for incoming messages
        threading.Thread(target=self.listen_for_messages, daemon=True).start()
    
    def connect_to_peer(self, peer_ip, port=5555):
        """Connect to another device as client"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"\nConnecting to {peer_ip}:{port}...")
        
        try:
            self.socket.connect((peer_ip, port))
            self.connected = True
            self.peer_address = (peer_ip, port)
            
            # Exchange usernames
            data = self.socket.recv(4096).decode('utf-8')
            message = json.loads(data)
            self.peer_username = message.get('username', 'Unknown')
            
            self.socket.send(json.dumps({'type': 'handshake', 'username': self.username}).encode('utf-8'))
            
            print(f"✓ Connected to {self.peer_username} at {peer_ip}:{port}")
            self.speak(f"Connection established with {self.peer_username}")
            
            # Start listening for incoming messages
            threading.Thread(target=self.listen_for_messages, daemon=True).start()
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            self.connected = False
    
    def listen_for_messages(self):
        """Listen for incoming payments"""
        while self.connected:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                message = json.loads(data)
                if message['type'] == 'payment':
                    self.handle_received_payment(message)
            except Exception as e:
                print(f"Error receiving message: {e}")
                self.connected = False
                break
    
    def handle_received_payment(self, message):
        """Process received payment"""
        amount = message['amount']
        sender = message['sender']
        timestamp = message['timestamp']
        transaction_id = message['transaction_id']
        
        self.balance += amount
        self.db.update_balance(self.username, self.balance)
        
        # Save transaction to database
        self.db.add_transaction(transaction_id, sender, self.username, amount, timestamp, 'completed')
        
        print(f"\n{'='*50}")
        print(f"💰 PAYMENT RECEIVED")
        print(f"From: {sender}")
        print(f"Amount: ₹{amount:,.2f}")
        print(f"New Balance: ₹{self.balance:,.2f}")
        print(f"Time: {timestamp}")
        print(f"{'='*50}\n")
        
        # TTS announcement
        self.speak(f"Payment received. {sender} sent you {amount} rupees. Your new balance is {self.balance:.2f} rupees")
        
        # Send confirmation
        confirmation = {
            'type': 'confirmation',
            'message': f'Payment of ₹{amount:,.2f} received successfully',
            'receiver_balance': self.balance
        }
        self.socket.send(json.dumps(confirmation).encode('utf-8'))
    
    def send_payment(self, amount):
        """Send payment to connected peer"""
        if not self.connected:
            print("✗ Not connected to any peer!")
            return False
        
        if amount <= 0:
            print("✗ Amount must be greater than 0")
            return False
        
        if amount > self.balance:
            print(f"✗ Insufficient balance! Current balance: ₹{self.balance:,.2f}")
            self.speak(f"Payment failed. Insufficient balance")
            return False
        
        # Generate unique transaction ID
        transaction_id = f"TXN{uuid.uuid4().hex[:12].upper()}"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Deduct from balance
        self.balance -= amount
        self.db.update_balance(self.username, self.balance)
        
        # Save transaction to database
        self.db.add_transaction(transaction_id, self.username, self.peer_username, amount, timestamp, 'completed')
        
        # Create payment message
        payment = {
            'type': 'payment',
            'amount': amount,
            'sender': self.username,
            'timestamp': timestamp,
            'transaction_id': transaction_id
        }
        
        try:
            self.socket.send(json.dumps(payment).encode('utf-8'))
            
            print(f"\n{'='*50}")
            print(f"💸 PAYMENT SENT")
            print(f"To: {self.peer_username}")
            print(f"Amount: ₹{amount:,.2f}")
            print(f"New Balance: ₹{self.balance:,.2f}")
            print(f"Transaction ID: {transaction_id}")
            print(f"{'='*50}\n")
            
            # TTS announcement
            self.speak(f"Payment sent. You paid {amount} rupees to {self.peer_username}. Your remaining balance is {self.balance:.2f} rupees")
            
            return True
        except Exception as e:
            # Refund if sending fails
            self.balance += amount
            self.db.update_balance(self.username, self.balance)
            print(f"✗ Payment failed: {e}")
            self.speak(f"Payment failed")
            return False
    
    def check_balance(self):
        """Check current balance"""
        print(f"\n💰 Current Balance: ₹{self.balance:,.2f}\n")
        self.speak(f"Your current balance is {self.balance:.2f} rupees")
    
    def view_transaction_history(self):
        """View transaction history"""
        transactions = self.db.get_user_transactions(self.username, limit=20)
        
        if not transactions:
            print("\n📋 No transactions found.\n")
            return
        
        print(f"\n{'='*90}")
        print(f"{'TRANSACTION HISTORY':^90}")
        print(f"{'='*90}")
        print(f"{'ID':<15} {'Date/Time':<20} {'From':<15} {'To':<15} {'Amount':<15} {'Status':<10}")
        print(f"{'-'*90}")
        
        for txn in transactions:
            txn_id, sender, receiver, amount, timestamp, status = txn
            
            # Highlight sent vs received
            if sender == self.username:
                amount_str = f"📤 -₹{amount:,.2f}"
            else:
                amount_str = f"📥 +₹{amount:,.2f}"
            
            print(f"{txn_id:<15} {timestamp:<20} {sender:<15} {receiver:<15} {amount_str:<15} {status:<10}")
        
        print(f"{'='*90}\n")
    
    def view_statistics(self):
        """View transaction statistics"""
        stats = self.db.get_transaction_stats(self.username)
        
        print(f"\n{'='*50}")
        print(f"{'TRANSACTION STATISTICS':^50}")
        print(f"{'='*50}")
        print(f"Total Sent:         ₹{stats['total_sent']:,.2f}")
        print(f"Total Received:     ₹{stats['total_received']:,.2f}")
        print(f"Net Flow:           ₹{stats['net_flow']:,.2f}")
        print(f"Total Transactions: {stats['transaction_count']}")
        print(f"Current Balance:    ₹{self.balance:,.2f}")
        print(f"{'='*50}\n")
    
    def close(self):
        """Close connection and database"""
        if self.socket:
            self.socket.close()
        self.connected = False
        self.db.close()


# ============================================
# MAIN PROGRAM
# ============================================

def main():
    print("="*50)
    print("     REAL-TIME PAYMENT SYSTEM")
    print("="*50)
    
    username = input("Enter your username: ")
    
    payment_system = PaymentSystem(username, initial_balance=10000)
    
    print("\nChoose mode:")
    print("1. Start as Server (wait for connection)")
    print("2. Connect to Server (enter peer IP)")
    
    mode = input("Enter choice (1 or 2): ")
    
    if mode == '1':
        payment_system.start_server(port=5555)
    else:
        peer_ip = input("Enter peer IP address: ")
        payment_system.connect_to_peer(peer_ip, port=5555)
    
    # Main menu
    while payment_system.connected:
        print("\n" + "="*50)
        print("MENU:")
        print("1. Send Payment")
        print("2. Check Balance")
        print("3. View Transaction History")
        print("4. View Statistics")
        print("5. Exit")
        print("="*50)
        
        choice = input("Enter choice: ")
        
        if choice == '1':
            try:
                amount = float(input("Enter amount to send (₹): "))
                payment_system.send_payment(amount)
            except ValueError:
                print("✗ Invalid amount")
        elif choice == '2':
            payment_system.check_balance()
        elif choice == '3':
            payment_system.view_transaction_history()
        elif choice == '4':
            payment_system.view_statistics()
        elif choice == '5':
            print("Closing connection...")
            payment_system.close()
            break
        else:
            print("✗ Invalid choice")
    
    print("Payment system closed.")


if __name__ == "__main__":
    main()
