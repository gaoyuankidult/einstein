import soket
import time

port = "5556"
server = soket.SoketServer(port)
while True:
    server.receive()
    time.sleep(1)