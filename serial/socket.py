import zmq

class Socket(object):

    def __init__(self, port="5556"):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)


    def send(self, msg):
        self.socket.send(msg)

    def receive(self):
        msg = self.socket.recv()
        return msg


class SocketServer(Socket):

    def __init__(self, port="5556"):
        super(SocketServer, self).__init__(port)
        self.socket.bind("tcp://*:%s" % port)

    def send_int(self, i):
        assert isinstance(i, int)
        self.send("%i\0" % i)


class SocketClient(Socket):

    def __init__(self, port="5556"):
        super(SocketClient, self).__init__(port)
        self.socket.connect("tcp://localhost:%s" % port)