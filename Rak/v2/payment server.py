import socket
import threading
import json
import pyttsx3
from datetime import datetime

class PaymentSystem:
    def __init__(self, username, initial_balance=1000):
        self.username = username
        self.balance = initial_balance
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        self.socket = None
        self.connected = False
        self.peer_address = None
        
    def speak(self, text):
        """Text-to-speech announcement"""
        print(f"[ANNOUNCEMENT] {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
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
        print(f"✓ Connected to {self.peer_address}")
        self.speak(f"Connection established")
        
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
            print(f"✓ Connected to {peer_ip}:{port}")
            self.speak(f"Connection established")
            
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
                self.handle_received_payment(message)
            except Exception as e:
                print(f"Error receiving message: {e}")
                self.connected = False
                break
    
    def handle_received_payment(self, message):
        """Process received payment"""
        if message['type'] == 'payment':
            amount = message['amount']
            sender = message['sender']
            timestamp = message['timestamp']
            
            self.balance += amount
            
            print(f"\n{'='*50}")
            print(f"💰 PAYMENT RECEIVED")
            print(f"From: {sender}")
            print(f"Amount: ${amount:.2f}")
            print(f"New Balance: ${self.balance:.2f}")
            print(f"Time: {timestamp}")
            print(f"{'='*50}\n")
            
            # TTS announcement
            self.speak(f"Payment received. {sender} sent you {amount} dollars. Your new balance is {self.balance:.2f} dollars")
            
            # Send confirmation
            confirmation = {
                'type': 'confirmation',
                'message': f'Payment of ${amount:.2f} received successfully',
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
            print(f"✗ Insufficient balance! Current balance: ${self.balance:.2f}")
            self.speak(f"Payment failed. Insufficient balance")
            return False
        
        # Deduct from balance
        self.balance -= amount
        
        # Create payment message
        payment = {
            'type': 'payment',
            'amount': amount,
            'sender': self.username,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            self.socket.send(json.dumps(payment).encode('utf-8'))
            
            print(f"\n{'='*50}")
            print(f"💸 PAYMENT SENT")
            print(f"Amount: ${amount:.2f}")
            print(f"New Balance: ${self.balance:.2f}")
            print(f"{'='*50}\n")
            
            # TTS announcement
            self.speak(f"Payment sent. You paid {amount} dollars. Your remaining balance is {self.balance:.2f} dollars")
            
            return True
        except Exception as e:
            # Refund if sending fails
            self.balance += amount
            print(f"✗ Payment failed: {e}")
            self.speak(f"Payment failed")
            return False
    
    def check_balance(self):
        """Check current balance"""
        print(f"\n💰 Current Balance: ${self.balance:.2f}\n")
        self.speak(f"Your current balance is {self.balance:.2f} dollars")
    
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
        self.connected = False


def main():
    print("="*50)
    print("     REAL-TIME PAYMENT SYSTEM")
    print("="*50)
    
    username = input("Enter your username: ")
    initial_balance = float(input("Enter initial balance (default 1000): ") or 1000)
    
    payment_system = PaymentSystem(username, initial_balance)
    
    print("\nChoose mode:")
    print("1. Start as Server (wait for connection)")
    print("2. Connect to Server (enter peer IP)")
    
    mode = input("Enter choice (1 or 2): ")
    
    if mode == '1':
        port = int(input("Enter port (default 5555): ") or 5555)
        payment_system.start_server(port=port)
    else:
        peer_ip = input("Enter peer IP address: ")
        port = int(input("Enter port (default 5555): ") or 5555)
        payment_system.connect_to_peer(peer_ip, port)
    
    # Main menu
    while payment_system.connected:
        print("\n" + "="*50)
        print("MENU:")
        print("1. Send Payment")
        print("2. Check Balance")
        print("3. Exit")
        print("="*50)
        
        choice = input("Enter choice: ")
        
        if choice == '1':
            try:
                amount = float(input("Enter amount to send: $"))
                payment_system.send_payment(amount)
            except ValueError:
                print("✗ Invalid amount")
        elif choice == '2':
            payment_system.check_balance()
        elif choice == '3':
            print("Closing connection...")
            payment_system.close()
            break
        else:
            print("✗ Invalid choice")
    
    print("Payment system closed.")


if __name__ == "__main__":
    main()
