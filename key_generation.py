import base64
import numpy as np
import binascii

from qiskit import QuantumCircuit,Aer,execute


def generate_key(n):
    np.random.seed(42)
    circ = QuantumCircuit(1,1)
    circ.x(0)
    circ.barrier()
    circ.h(0)
    circ.barrier()
    circ.measure(0,0)
    circ.barrier()
    circ.draw(output='text')
    backend = Aer.get_backend('qasm_simulator')
    result = execute(circ, backend, shots=n,
    memory = True).result()
    bits_alice = [int(q) for q in result.get_memory()]
    #print("la longueur de bits générés par Alice : ",len(bits_alice))
    #print("Bits générés par Alice : ",bits_alice)
    Alice_binary = []
    for i in bits_alice:
        Alice_binary.append(str(i))

    result = execute(circ, backend, shots=n,
    memory = True).result()
    basis_bob = [int(q) for q in result.get_memory()]
    #print("lla longueur de bits générés par Bob :",len(basis_bob))
    #print("Bits générés par Bob :",basis_bob)
    bob_base_binary = []
    for i in basis_bob:
        bob_base_binary.append(str(i))

    bits_bob = []
    for i in range(n):
        circ_send = QuantumCircuit(1,1)
        if bits_alice[i] == 0:
            circ_send.id(0)
        if bits_alice[i] == 1:
            circ_send.h(0)
        else:
            circ_send.id(0)
        circ_send.measure(0,0)
        result = execute(circ_send, backend, shots = 1,
    memory = True).result()
        bits_bob.append(int(result.get_memory()[0]))
    #print("longeur de bits après mésure :",len(bits_bob))
    #print("bits mésurés par Bob :",bits_bob)
    bob_binary = []
    for i in bits_bob:
        bob_binary.append(str(i))
    key = []
    for i in range(n):
        if bits_alice[i] == bits_bob[i]:
            key.append(bits_bob[i])
    #print("longueur la clé:", len(key))
    #print("la clé secrète :", key)
    key_final=[]
    for i in key:
        key_final.append(str(i))
    return Alice_binary,bob_binary,bob_base_binary,key_final
bits_alice,bits_bob,basis_bob,key=generate_key(100)
#print(bits_alice)
def toString(the_string):
    str1=""
    for i in the_string:
        str1+=i
    return str1
my_string=toString(bits_alice)
#print(" io : ",int(my_string))

def toDecimal(bin):
    last_string=int(bin,2)
    return last_string
dec=toDecimal(my_string)
#print(dec)
def the_final_key(the_key):
    strTwo=""
    for i in range (0,len(the_key),7):
        tempo=the_key[i:i+7]
        decimal_data=toDecimal(tempo)
        strTwo=strTwo+chr(decimal_data)
    return strTwo
final=the_final_key(my_string)
#print(final)

