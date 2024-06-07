import numpy as np
import heapq
import pickle
from itertools import zip_longest, islice
from PIL import Image
from collections import Counter
from collections import namedtuple
from decimal import Decimal
from math import *
import sys
import os
from tkinter import W
import turtle
symb = 'ъ'
window = 15
def entrop(ar):
    ent = 0
    for x in ar:
        ent-=x*log2(x)
    return ent
def rle(s):
    encoded = []
    count = 1
    flag = chr(256)
    strlen = len(s)
    for i in range(1, strlen):
        if s[i] == s[i - 1]:
            count += 1
        else:
            if count == 1:
                encoded.append(s[i - 1])
            else:
                encoded.append(count)
                encoded.append(flag)
                encoded.append(s[i - 1])
            count = 1
    if count == 1:
        encoded.append(s[len(s) - 1])
    else:
        encoded.append(count)
        encoded.append(flag)
        encoded.append(s[len(s) - 1])
    res = ''
    for i in encoded:
        if (i==flag):
            res += flag

        else:
            res += str(i) + ' '
    return res
def RLEimg(s):
    k = 1
    l = []
    for i in range(1,len(s)):
        if (np.array_equal(s[i],s[i-1])):
            k+=1
        else:
            m = [k,s[i-1]]
            l.append(m)
            k = 1
    l.append(m)
    return l
def RLEback(s):
    l = []
    for i in range(len(s)):
        if (s[i].isdigit()):
            k=int(s[i])
            for j in range(k-1):
                l.append(s[i+1])
        else:
            l.append(s[i])
    return "".join(l)
def imgtoraw(img):
    imgar = np.array(img)
    #imgar = np.mean(imgar,axis=1)
    imgar = imgar.astype(np.uint8)
    return imgar
def to_int_keys_best(l):
    seen = set()
    ls = []
    te = ''
    for e in reversed(l):
        te = e + te
        if not te in seen:
            ls.append(te)
            seen.add(te)
    ls = sorted(ls)
    befind = ls.index(l)
    index = [l.index(x) for x in ls]
    return [index, befind]

def BWT(s):
    s = s + symb
    n = len(s)
    k = 1
    code = []
    line = to_int_keys_best(s)
    for x in line[0]:
        code.append(s[x-1])
    return [code, line[1]]
    # s.sort()
    # ind = data.index(s)
    # l = "".join([data[i][-1] for i in range(len(data))])
    # return (l,ind)
def inverse_BWT(m):
    mas = m[0]
    ind = m[1]
    data = ['' for j in range(len(mas))]
    while(len(data[0])!=len(mas)):
        data.sort()
        data = [mas[i]+data[i] for i in range(len(mas))]
    data.sort()
    return (data[ind])
def avrept(s):
    l = ""
    col = 1
    count = 0
    for i in range(1,len(s)):
        if (s[i]==s[i-1]):
            col+=1
        else:
            if (col>1):
                count+=1
                l+=s[i-1]*col
                col=1
    return (len(l)-2*count)/len(s)
def MTF(s):
    res = []
    alp = ''.join(sorted(set(s), key=lambda x: x.lower())) #если так, то алфавит нужен
    #alp = ''.join([chr(x) for x in range(97,123)]) #так алфавит не нужен
    for x in s:
        ind = alp.index(x)
        res.append(str(ind))
        alp = alp[ind] + alp[:ind] + alp[ind+1:]
    return [res,alp]
def inv_MTF(a,b):
    alp = ''.join(sorted(list(b)))
    s = []
    for x in a:
        s.append(alp[int(x)])
        alp = alp[int(x)] + alp[:int(x)] + alp[int(x) + 1:]
    return ''.join(s)
def probab(s):
    n = sorted(set(s), key=lambda x: x.lower())
    d = []
    for x in n:
        coun = 0
        for i in s:
            if (i == x):
                coun+=1
        d.append(coun/len(s))
    return d
def arifm(s,d):
    mas = [0]*(len(d)+1)
    mas[0] = 0
    coun=0
    mas[len(d)] = 1
    i = 1
    for x in d:
        mas[i] = d[x] + mas[i-1]
        i+=1
    l = list(d.keys())
    for n in s:
        if (coun!=len(s)-1):
            mas[len(d)] = mas[l.index(n)+1]
            mas[0] = mas[l.index(n)]
            for i in range(1,len(d)):
                mas[i] = mas[i-1] + (mas[len(d)]-mas[0])*d[l[i-1]]
            coun+=1
    return [(mas[l.index(n)] + mas[l.index(n)+1])/2,coun+1]

def inv_arifm(n,len,d):
    res = []
    left = 0
    right = 1
    l = list(d.keys())
    d = list(d.values())
    for _ in range(len):
        for i, letter in enumerate(l):
            interval = (left + (right - left) * sum(d[:i]),
                        left + (right - left) * sum(d[:i + 1]))
            if interval[0] <= n < interval[1]:
                res.append(letter)
                n = (n - interval[0]) / (interval[1] - interval[0])
                break

    return ''.join(res)
def reverseBWT(m):
    mas = list(m[0])
    firs = sorted(mas)
    ind = m[1]
    left = [x for x in range(len(mas))]
    right = [0]*len(mas)

    alph = list(set(m[0]))
    cont = {}
    cont = dict.fromkeys(alph, 0)
    for i in range(len(mas)):
        cont[mas[i]] += 1
    j=0
    for x in mas:
        right[j] = firs.index(x)+(firs.count(x)-cont[x])
        cont[x]-=1
        j+=1
    '''  
    alph = list(set(m[0]))
    count = {}
    count = dict.fromkeys(alph, 0)
    for i in range(len(mas)):
        count[mas[i]]+=1
    sum = 0
    
    for i in range(len(alph)):
        sum = sum + count[mas[i]]
        count[mas[i]] = sum - count[mas[i]]
    t = [0]*len(mas)
    for i in range(len(mas)):
        print(count[mas[i]])
        t[count[mas[i]]]=i
        count[mas[i]]+=1
    '''
    j = right[ind]
    ans = [0] * len(mas)
    for i in range(len(mas)):
        ans[i] = firs[left[j]]
        j = right[j]
    ans.reverse()
    return ans
def count_probability(data):  # find probability for Arithmetic coding
    num = 0
    sr = ""
    dictionary = []
    for i in data:
        if (sr.count(i) == 0):
            k = data.count(i)
            num += k
            dictionary.append((i, k))
            sr += i
    dictionary.sort(key=lambda x: x[1], reverse=True)

    probability = []
    indices = {}
    k = 0

    for i in range(len(dictionary)):
        probability.append(dictionary[i][1] / num)
        indices[dictionary[i][0]] = k
        k += 1

    return probability, indices


def read_probability(decode):
    file = open("Arithmetic_probability.txt", "r", encoding="utf-8")  # get probability from txt
    data = file.read()
    probability = []
    indices = {}
    sr = ""
    for i in data[6:]:
        if (i != chr(2)):
            sr += i
        else:
            break
    num = int(sr)
    ln = len(sr)
    sr = ""
    n = False
    k = 0
    sym = ''

    for i in data[7 + ln:]:
        if (i == '='):
            n = True
        elif (i != chr(2) and n):
            sr += i
        elif (i == chr(2)):
            n = False
            probability.append(int(sr) / num)
            sr = ""
            if (decode):
                indices[k] = sym
                k += 1
        else:
            if (decode):
                sym = i
            else:
                indices[i] = k
                k += 1
    return probability, indices

def arithmetic():  # Arithmetic coding for double
    file = open("C:/Users/user/Desktop/text.txt", "r", encoding="utf-8")
    file2 = open("C:/Users/user/Desktop/textout.txt", "w", encoding="utf-8")
    data = file.read()
    probability, indices = count_probability(data)
    intervals = [sum(probability[:i]) for i in range(len(probability) + 1)]
    Left = 0
    Right = 1
    length = Right - Left
    num = 0
    double_list = []
    num_list = []

    print("")
    i = 0
    while i < len(data):
        length = Right - Left
        Left, Right = Left + intervals[indices[data[i]]] * length, Left + intervals[indices[data[i]] + 1] * length
        print(str(num) + " " + str(Left) + " " + str(Right))
        for j in range(len(indices)):
            L, R = Left + intervals[j] * length, Left + intervals[j + 1] * length
            if (L == R):
                print("Error may occur")
                double_list.append(tmp)
                Left = 0
                Right = 1
                num_list.append(num)
                num = -1
                i -= 1
                break
        i += 1
        num += 1
        tmp = (Left + Right) / 2
    double_list.append((Left + Right) / 2)
    num_list.append(num)
    print("\n")
    for i in range(len(double_list)):
        print(str(num_list[i]) + " " + str(double_list[i]))
        file2.write(str(num_list[i]) + "," + str(double_list[i]) + ",")
    file.close()
    file2.close()


def decode_arithmetic():
    file = open("textout.txt", "r", encoding="utf-8")  # decode Arithmetic coding
    file2 = open("text.txt", "w", encoding="utf-8")

    probability, indices = read_probability(True)
    sr = ""
    double_list = []
    num_list = []
    data = file.read()

    nm = True
    for i in data:
        if (i != ","):
            sr += i
        elif (nm):
            num_list.append(int(sr))
            sr = ""
            nm = False
        else:
            double_list.append(Decimal(sr))
            sr = ""
            nm = True

    intervals = [sum(probability[:i]) for i in range(len(probability) + 1)]
    data2 = ""
    for c in range(len(double_list)):
        Left = 0
        Right = 1
        number = double_list[c]
        num = num_list[c]
        j = 0
        changing_intervals = intervals

        while (Left != Right and j < num):
            index_interval = sum([number > i for i in changing_intervals]) - 1
            data2 = data2 + indices[index_interval]
            Left = changing_intervals[index_interval]
            Right = changing_intervals[index_interval + 1]
            length = Right - Left
            changing_intervals = [Left + length * intervals[i] for i in range(len(intervals))]
            print(changing_intervals)
            j += 1

    print("\n" + data2)
    file2.write(data2)
    file.close()
    file2.close()

def huffmandec(encoded, code):
    sx =[]
    enc_ch = ""
    for ch in encoded:
        enc_ch += ch
        for dec_ch in code:
            if code.get(dec_ch) == enc_ch:
                sx.append(dec_ch)
                enc_ch = ""
                break
    return "".join(sx)
class Node(namedtuple("Node", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")
class Leaf(namedtuple("Leaf", ["char"])):
    def walk(self, code, acc):
        code[self.char] = acc or "0"
def huffman(s):
    h = []
    for ch, freq in Counter(s).items():
        h.append((freq, len(h), Leaf(ch)))
    heapq.heapify(h)
    count = len(h)
    while len(h) > 1:
        freq1, _count1, left = heapq.heappop(h)
        freq2, _count2, right = heapq.heappop(h)
        heapq.heappush(h, (freq1 + freq2, count, Node(left, right)))
        count += 1
    code = {}
    if h:
        [(_freq, _count, root)] = h
        root.walk(code, "")
    return code
def findMatch(buf, pos,s):
    flag = 0
    i = 0
    l = s[pos]
    length = offset = 0
    while flag!=-1:
        flag = buf.find(l)
        length = i
        if (flag!=-1):
            offset = pos - flag
        i+=1
        l += s[pos + i]
    return offset, length
def find_longest_match(data, current_position):
    end_of_buffer = min(current_position + 258, len(data))
    best_match_distance = -1
    best_match_length = -1

    for j in range(current_position - 1, max(current_position - 255, -1), -1):
        match_length = 0
        while (match_length < (end_of_buffer - current_position) and
               current_position + match_length < len(data) and
               j + match_length < len(data) and
               data[j + match_length] == data[current_position + match_length]):
            match_length += 1

        if match_length > best_match_length:
            best_match_distance = current_position - j
            best_match_length = match_length

    return best_match_distance, best_match_length

def lz77_encode(data):
    i = 0
    output = []
    while i < len(data):
        match_distance, match_length = find_longest_match(data, i)

        if match_distance > 0 and match_length > 0 and i + match_length < len(data):
            output.append([match_distance, match_length, data[i + match_length]])
            i += match_length + 1
        else:
            output.append([0, 0, data[i]])
            i += 1

    return [(item[0], item[1], item[2]) for item in output]

def lz77_decode(encoded_data):
    output = []
    for offset, length, symbol in encoded_data:
        if offset == 0 and length == 0:
            output.append(symbol)
        else:
            for j in range(length):
                output.append(output[-offset])
            output.append(symbol)
    return ''.join(output)

def lz77_compress(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    encoded_data = lz77_encode(data)
    serialized_data = ''.join([chr(x[0]) + chr(x[1]) + x[2] for x in encoded_data])
    with open(output_path, 'wb') as f:
        f.write(serialized_data.encode('utf-8', 'surrogatepass'))

def lz77_decompress(input_path):
    with open(input_path, 'rb') as f:
        data = f.read().decode('utf-8', 'surrogatepass')
    serialized_data = [(ord(data[i]), ord(data[i + 1]), data[i + 2]) for i in range(0, len(data), 3)]
    return lz77_decode(serialized_data)
# f = 'C:/Users/user/Desktop/text.txt'
# g = 'C:/Users/user/Desktop/textout.txt'
# lz77_compress(f,g)
# #
#
# arithmetic()
#
# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'wb')
# n = f.readlines()
# m = ''
# for x in n:
#     bwt = BWT(x)
#     m += "".join(bwt[0]) + ' ' + str(bwt[1])
# code = huffman(m)  # кодируем строку
# encoded = ''.join(code[ch] for ch in m)
# rest = len(encoded) % 8
# encoded = '0' * rest + encoded
# n = int(encoded, 2)
# n = n.to_bytes(len(encoded)//4, "big")
# g.write(n)
# f.close()
# g.close()

# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'w')
# n = f.readlines()
# i = 0
# m = []
# for x in n:
#     bwt = BWT(x)
#     mtf = MTF("".join(bwt[0]) + symb + str(bwt[1]))
#     m.append("".join(bwt[0]) + symb + str(bwt[1]))
# for x in m:
#     code = rle(x)  # кодируем строку
#     g.write(code)
# f.close()
# g.close()

# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'wb')
# n = f.readlines()
# i = 0
# m = ""
# for x in n:
#     bwt = BWT(x)
#     mtf = MTF("".join(bwt[0]) + symb + str(bwt[1]))
#     m+=("".join(mtf[0]))
# code = huffman(m)  # кодируем строку
# encoded = ''.join(code[ch] for ch in m)
# rest = len(encoded) % 8
# encoded = '0' * rest + encoded
# m = int(encoded, 2)
# m = m.to_bytes(len(encoded)//4, "big")
# g.write(m)
# f.close()
# g.close()

f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
g = open("C:/Users/user/Desktop/textout.txt", 'w')
n = f.readline()
s = huffman(n)
encoded = ''.join(s[ch] for ch in n)
mas = []
probb = probab(n)
print(entrop(probb))
print(len(encoded)/len(n))
g.write(encoded)

# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.bin", 'wb')
# n = f.readlines()
# i = 0
# s = ''
# for x in n:
#     bwt = BWT(x)
#     mtf = ''.join(MTF("".join(bwt[0]) + symb + str(bwt[1]))[0])
#     s = s + rle(mtf)
# code = huffman(s)  # кодируем строку
# encoded = ''.join(code[ch] for ch in s)
# rest = len(encoded) % 8
# encoded = '0' * rest + encoded
# s = int(encoded, 2)
# y = s.to_bytes(len(encoded)//4, "big")
# g.write(y)
# f.close()
# g.close()

#
# img = Image.open("C:/Users/user/PycharmProjects/pythonProject/nelena.jpg")
# img = img.convert("RGB")
# np.set_printoptions(threshold=np.inf)
# arr = imgtoraw(img)
# arr1 = np.reshape(arr,(len(arr),len(arr[0])*3))
# arr = np.reshape(arr,(len(arr)*len(arr[0]),3))
# arr = np.uint(arr)
# np.savetxt("C:/Users/user/Desktop/lab1.txt", arr1, fmt="%d")
# R = [x[0] for x in arr]
# G = [x[1] for x in arr]
# B = [x[2] for x in arr]
# g = open("C:/Users/user/Desktop/outlab1.txt", 'wb')
# f = rle(R)
# s = rle(G)
# t = rle(B)
# ff = len(f).to_bytes(4, byteorder='big')
# ss = len(s).to_bytes(4, byteorder='big')
# tt = len(t).to_bytes(4, byteorder='big')
# sr = f+s+t
# sr = sr.encode('utf-8')
# sr = sr + ff + ss + tt
# g.write(sr)
# g.close()
# img.close()
# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'w', encoding='utf-8')
# n = f.readlines()
# for x in n:
#     g.write(rle(x))
# f.close()
# g.close()
