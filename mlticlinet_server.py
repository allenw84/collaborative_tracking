from socket import *
import threading
import pickle
from time import sleep 
from demo_yolo3_deepsort import Detector

def findOverlapUser(overlap,user_entry_dict):
    userid = -1
    for entry in list(user_entry_dict.values()):
        for grid in overlap:
            if grid not in entry[3]:
                continue
        userid = list(user_entry_dict.keys())[list(user_entry_dict.values()).index(entry)]
        print('ss',userid[0])
        #break;
    return userid

def handleOverlap(entry, msg):
    for new_feature in msg[1]:
        for old_feature in entry[0]:
            if new_feature[0] == old_feature[0] & new_feature[1] > old_feature[1]:
                old_feature[1] = new_feature[1]
                old_feature[2] = new_feature[2]
    
    entry[2].append(msg[3]) 
        

def rev(tcpclientsocket,addr, user_entry_dict):
    try:
        data = tcpclientsocket.recv(1024)
        recv_msg = pickle.loads(data)
        print(recv_msg)
        suc='successful'
        tcpclientsocket.send(suc.encode())
        
    except:
        print(addr,' leaves')
        return
    
    overlap_userid = findOverlapUser(recv_msg[2],user_entry_dict)
    print(overlap_userid)
    handleOverlap(user_entry_dict[overlap_userid], recv_msg)

def send(tcpclientsocket,addr):
    try:
        suc='Hello,I am server. '
        tcpclientsocket.send(suc.encode())
        sleep(9)
    except:
        print(addr,'Unconnected')
        return
def get_entry_dict():
    entry  = Detector.return_user_dict() 
    print(entry)
    return entry
HOST = ''
PORT = 10523
ADDR = (HOST,PORT)
server_socket = socket(AF_INET,SOCK_STREAM)
server_socket.bind(ADDR)
server_socket.listen(5)
print ('Waiting for connecting')
flag = 0
user_entry_dict = get_entry_dict()
while True:
    print(flag)
    if flag == 1000: #checking loop times
        user_entry_dict = get_entry_dict()
        flag = 0
    tcpclientsocket,addr = server_socket.accept()
    print('\n')
    print( 'Connected by ',addr)
    threading.Thread(target=rev,args=(tcpclientsocket,addr,user_entry_dict)).start()
    flag+=1
