import soket
import time

port = "5556"
client = soket.SoketClient(port)
while True:
    client.send("I greet you.")