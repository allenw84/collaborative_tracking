from socket import *
import threading
import pickle
from time import sleep 
import demo_yolo3_deepsort
from demo_yolo3_deepsort import Detector

global user_entry
user_entry = {}

def findOverlapUser(overlap):
    global user_entry
    userid = -1
    #print(overlap)
    #print(user_entry)
    for entry in list(user_entry.values()):
        for grid in overlap:
            if grid not in entry[3]:
                continue
        userid = list(user_entry.keys())[list(user_entry.values()).index(entry)]
        #print('ss',userid)
        #break;
    return userid

def handleOverlap(entry, msg):
    for new_feature in msg[1]:
        for old_feature in entry[0]:
            if new_feature[0] == old_feature[0] and new_feature[1] > old_feature[1]:
                old_feature[1] = new_feature[1]
                old_feature[2] = new_feature[2]
    
    entry[2] += (msg[3]) 
        

def rev(tcpclientsocket,addr):
    global user_entry

    try:
        data = tcpclientsocket.recv(1024)
        recv_msg = pickle.loads(data)
        #print("recv", recv_msg)
        suc='successful'
        tcpclientsocket.send(suc.encode())
        if isinstance(recv_msg, dict):
            user_entry = recv_msg
            #print(user_entry)
            return
        
    except:
        print(addr,' leaves')
        return
    #print("msg")
    #print(user_entry)
    overlap_userid = findOverlapUser(recv_msg[2])
    #print(overlap_userid)
    handleOverlap(user_entry[overlap_userid], recv_msg)

def send(tcpclientsocket,addr):
    try:
        suc='Hello,I am server. '
        tcpclientsocket.send(suc.encode())
        sleep(9)
    except:
        print(addr,'Unconnected')
        return
HOST = ''
PORT = 10523
ADDR = (HOST,PORT)
server_socket = socket(AF_INET,SOCK_STREAM)
server_socket.bind(ADDR)
server_socket.listen(5)
print ('Waiting for connecting')

#user_entry_dict = get_entry_dict()
#user_entry = demo_yolo3_deepsort.user_entry_dict
while True:
    tcpclientsocket,addr = server_socket.accept()
    print('\n')
    print( 'Connected by ',addr)
    threading.Thread(target=rev,args=(tcpclientsocket,addr)).start()
