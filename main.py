import network
import socket
from machine import Pin, PWM
from time import sleep

# Wi-Fi Credentials
SSID = "LIBRARY_05"
PASSWORD = "library_05"

# Connect to Wi-Fi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

print("Connecting to Wi-Fi...")
while not wlan.isconnected():
    pass

print("Connected! IP:", wlan.ifconfig()[0])

# Setup Servo on GPIO 15
servo = PWM(Pin(15))
servo.freq(50)  # 50Hz for SG90

# Function to move servo
def set_angle(angle):
    min_duty = 1638  # 0° (0.5ms)
    max_duty = 8192  # 180° (2.5ms)
    duty = int(min_duty + (angle / 180) * (max_duty - min_duty))
    servo.duty_u16(duty)

# Start HTTP server
def start_server():
    addr = ('0.0.0.0', 5500)
    s = socket.socket()
    s.bind(addr)
    s.listen(1)
    print("Server listening on http://0.0.0.0:5500")

    while True:
        conn, addr = s.accept()
        print("Client connected from:", addr)
        request = conn.recv(1024).decode()

        if "GET /clockwise" in request:
            print("Moving Right")
            set_angle(180)  # Move servo right
            response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nMoved Right"
        elif "GET /counterclockwise" in request:
            print("Moving Left")
            set_angle(0)  # Move servo left
            response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nMoved Left"
        else:
            response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nUnknown Command"

        conn.send(response)
        conn.close()

# Run server
start_server()
