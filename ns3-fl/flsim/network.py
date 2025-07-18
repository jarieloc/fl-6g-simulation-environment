#from py_interface import *
from ctypes import *
import socket
import struct
import subprocess

TCP_IP = '127.0.0.1'
TCP_PORT = 8080
PATH='../ns3-fl-network'
PROGRAM='wifi_exp'

class Network(object):
    def __init__(self, config):
        self.config = config
        self.num_clients = self.config.clients.total
        self.network_type = self.config.network.type

        proc = subprocess.Popen('./waf build', shell=True, stdout=subprocess.PIPE,
                                universal_newlines=True, cwd=PATH)
        proc.wait()
        if proc.returncode != 0:
            exit(-1)

        command = './waf --run "' + PROGRAM + ' --NumClients=' + str(self.num_clients) + ' --NetworkType=' + self.network_type
        command += ' --ModelSize=' + str(self.config.model.size)
        '''print(self.config.network)
        for net in self.config.network:
            if net == self.network_type:
                print(net.items())'''

        if self.network_type == 'wifi':
            command += ' --TxGain=' + str(self.config.network.wifi['tx_gain'])
            command += ' --MaxPacketSize=' + str(self.config.network.wifi['max_packet_size'])
        else: # else assume ethernet
            command += ' --MaxPacketSize=' + str(self.config.network.ethernet['max_packet_size'])

        command += " --LearningModel=" + str(self.config.server)

        command += '"'
        print(command)

        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                universal_newlines=True, cwd=PATH)


    def parse_clients(self, clients):
        clients_to_send = [0 for _ in range(self.num_clients)]
        for client in clients:
            clients_to_send[client.client_id] = 1
        return clients_to_send

    def connect(self):
        self.s = socket.create_connection((TCP_IP, TCP_PORT,))

    def sendRequest(self, *, requestType: int, array: list):
        print("sending")
        print(array)
        message = struct.pack("II", requestType, len(array))
        self.s.send(message)
        # for the total number of clients
        # is the index in lit at client.id equal
        for ele in array:
            self.s.send(struct.pack("I", ele))

        resp = self.s.recv(8)
        print("resp")
        print(resp)
        if len(resp) < 8:
            print(len(resp), resp)
        command, nItems = struct.unpack("II", resp)
        ret = {}
        for i in range(nItems):
            dr = self.s.recv(8 * 3)
            eid, roundTime, throughput = struct.unpack("Qdd", dr)
            temp = {"roundTime": roundTime, "throughput": throughput}
            ret[eid] = temp
        return ret

    def sendAsyncRequest(self, *, requestType: int, array: list):
        print("sending")
        print(array)
        message = struct.pack("II",requestType , len(array))
        self.s.send(message)
        # for the total number of clients
        # is the index in lit at client.id equal
        for ele in array:
            self.s.send(struct.pack("I", ele))

    def readAsyncResponse(self):
        resp = self.s.recv(8)
        print("resp")
        print(resp)
        if len(resp) < 8:
            print(len(resp), resp)
        command, nItems = struct.unpack("II", resp)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(command)
        if command == 3:
            return 'end'
        ret = {}
        for i in range(nItems):
            dr = self.s.recv(8 * 4)
            eid, startTime, endTime, throughput = struct.unpack("Qddd", dr)
            temp = {"startTime": startTime, "endTime": endTime, "throughput": throughput}
            ret[eid] = temp
        return ret


    def disconnect(self):
        # self.sendAsyncRequest(requestType=2, array=[])
        self.s.close()

