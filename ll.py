import numpy as np
import heapq
import pickle
from itertools import zip_longest, islice
from PIL import Image
from collections import Counter
from collections import namedtuple
from math import *
symb = 'ъ'
def entrop(ar):
    ent = 0
    for x in ar:
        ent-=x*log2(x)
    return entrop
def RLE(s):
    k = 1
    l = []
    for i in range(1,len(s)):
        if (s[i] == s[i-1]):
            k+=1
        else:
            if (k!=1):
                l.append(str(k))
            l.append(s[i-1])
            k = 1
    if (k!=1):
        l.append(str(k))
    l.append(s[len(s)-1])
    return ''.join(l)
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

#
#     data.sort()
#     ind = data.index(s)
#     l = "".join([data[i][-1] for i in range(len(data))])
#     return (l,ind)
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
def p(s):
    n = sorted(set(s), key=lambda x: x.lower())
    d = {}
    for x in n:
        coun = 0
        for i in s:
            if (i == x):
                coun+=1
        d[x] = coun/len(s)
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
def huffmandec(encoded, code):  # функция декодирования исходной строки по кодам Хаффмана
    sx =[]  # инициализируем массив символов раскодированной строки
    enc_ch = ""  # инициализируем значение закодированного символа
    for ch in encoded:  # обойдем закодированную строку по символам
        enc_ch += ch  # добавим текущий символ к строке закодированного символа
        for dec_ch in code:  # постараемся найти закодированный символ в словаре кодов
            if code.get(dec_ch) == enc_ch:  # если закодированный символ найден,
                sx.append(dec_ch)  # добавим значение раскодированного символа к массиву раскодированной строки
                enc_ch = ""  # обнулим значение закодированного символа
                break
    return "".join(sx)  # вернем значение раскодированной строки
class Node(namedtuple("Node", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")  # пойти в левого потомка, добавив к префиксу "0"
        self.right.walk(code, acc + "1")  # пойти в правого потомка, добавив к префиксу "1"
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

def longest_prefix_from(Left, Right):
    LongestPrefixLength = 0
    LongestPrefixPos = -1
    while 1:
        PrefixLength = LongestPrefixLength+1
        if PrefixLength >= len(Right):
            break
        Prefix = Right[0: PrefixLength]
        PrefixPos = Left.find(Prefix)
        if PrefixPos == -1:
            break
        LongestPrefixLength = PrefixLength
        LongestPrefixPos = PrefixPos
    return (LongestPrefixLength, LongestPrefixPos)
def codeBufferLZ77(Buffer):
    Result = []
    CodePos = 0
    while CodePos < len(Buffer):
        Left = Buffer[0:CodePos]
        Right = Buffer[CodePos:]
        (PrefixLength, PrefixPos) = longest_prefix_from(Left, Right)
        if (PrefixLength == 0):
            Result.append(Buffer[CodePos])
            CodePos = CodePos+1
        else:
            Result.append((PrefixLength, CodePos-PrefixPos))
            CodePos = CodePos + PrefixLength
    return Result
def lz77code(input_filename, output_filename, WindowSize):
    input_file = open(input_filename, 'r')
    CodedText = []
    while 1:
        Buffer = input_file.read(WindowSize)
        if Buffer == '':
            break
        Code = codeBufferLZ77(Buffer)
        CodedText += Code
    input_file.close()
    output_file = open(output_filename, 'wb')
    pickle.dump(CodedText, output_file)
    output_file.close()
def lz77decode(input_filename, output_filename):
    input_file = open(input_filename, 'rb')
    CodedList = pickle.load(input_file)
    DecodedText = ''
    Pos = 0
    for Value in CodedList:
        if isinstance(Value, str):
            DecodedText = DecodedText+Value
            Pos = Pos + 1
        else:
            (PrefixLength, Shift) = Value
            PrefixPos = Pos-Shift
            DecodedText += DecodedText[PrefixPos:PrefixPos+PrefixLength]
            Pos = Pos + PrefixLength
    output_file = open(output_filename, 'w')
    output_file.write(DecodedText)
    output_file.close()
print(longest_prefix_from('asddsafsg', 'asdjghmvbnv'))
print(codeBufferLZ77('ababgsdfghhsdafaasdasd'))
lz77code('in.txt', 'code.txt', 100)
lz77decode('code.txt', 'decode.txt')
#print(code_to_char)

# s = input()
# code = huffman(s)  # кодируем строку
# encoded = "".join(code[ch] for ch in s)  # отобразим закодированную версию, отобразив каждый символ
# print(huffmandec(encoded, code))
# # в соответствующий код и конкатенируем результат
# print(len(code), len(encoded))  # выведем число символов и длину закодированной строки
# for ch in sorted(code):  # обойдем символы в словаре в алфавитном порядке с помощью функции sorted()
#     print("{}: {}".format(ch, code[ch]))  # выведем символ и соответствующий ему код
# print(encoded)  # выведем закодированную строку

# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'w', encoding='utf-8')
# n = f.readlines()
# for x in n:
#     m = p(x)
#     g.write(str(m))
#     b = arifm(x,m)
#     g.write(str(b))
# f.close()
# g.close()
#
# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'wb')
# n = f.readlines()
# for x in n:
#     bwt = BWT(x)
#     x = "".join(bwt[0]) + symb + str(bwt[1])
# code = huffman(n)  # кодируем строку
# encoded = ''.join(code[ch] for ch in n)
# rest = len(encoded) % 8
# encoded = '0' * rest + encoded
# n = int(encoded, 2)
# n = n.to_bytes(len(encoded)//8, "big")
# g.write(n)
# f.close()
# g.close()


#  f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'w', encoding='utf-8')
# n = f.read()
# m = p(n)
# code = arifm(n,m)  # кодируем строку
# #encoded = "".join(code[ch] for ch in n)
# g.write(str(code))
# f.close()
# g.close()
#
# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'w', encoding='utf-8')
# n = f.readlines()
# i = 0
# m = []
# for x in n:
#     bwt = BWT(x)
#     mtf = MTF("".join(bwt[0]) + symb + str(bwt[1]))
#     m.append("".join(bwt[0]) + symb + str(bwt[1]))
#
# for x in m:
#     code = RLE(x)  # кодируем строку
# #encoded = "".join(code[ch] for ch in n)
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
# m = m.to_bytes(len(encoded)//8, "big")
# g.write(m)
# f.close()
# g.close()

# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'w', encoding='utf-8')
# n = f.readlines()
# i = 0
# m = ""
# for x in n:
#     bwt = BWT(x)
#     mtf = MTF("".join(bwt[0]) + symb + str(bwt[1]))
#     m+=("".join(mtf[0]))
# code = arifm(m,p(m))  # кодируем строку
# g.write(str(code))
# f.close()
# g.close()

# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'wb')
# n = f.readlines()
# i = 0
# s = ''
# for x in n:
#     bwt = BWT(x)
#     mtf = ''.join(MTF("".join(bwt[0]) + symb + str(bwt[1]))[0])
#     s = s + RLE(mtf)
# code = huffman(s)  # кодируем строку
# encoded = ''.join(code[ch] for ch in s)
# rest = len(encoded) % 8
# encoded = '0' * rest + encoded
# s = int(encoded, 2)
# y = s.to_bytes(len(encoded)//8, "big")
# g.write(y)
# f.close()
# g.close()

#
# f = open("C:/Users/user/Desktop/text.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/textout.txt", 'w', encoding='utf-8')
# n = f.readlines()
# i = 0
# s = ''
# for x in n:
#     bwt = BWT(x)
#     mtf = ''.join(MTF("".join(bwt[0]) + symb + str(bwt[1]))[0])
#     s = s + RLE(mtf)
# code = arifm(s,p(s))  # кодируем строку
# g.write(str(code))
# f.close()
# g.close()

# res = mas[alp]
# alp = mas[1]
# alp = ''.join(sorted(set(s), key=lambda x: x.lower()))
# for x in s:
#     ind = alp.index(x)
#     res+=str(ind)
#     alp = alp[ind] + alp[:ind] + alp[ind+1:]
# return [res, alp]


# img = Image.open("C:/Users/user/Desktop/1616975653_41-p-fon-v-kletku-cherno-belii-45.png")
# img = img.resize((100,100))
# img = img.convert("RGB")
# np.set_printoptions(threshold=np.inf)
# img.save("C:/Users/user/Desktop/new.png")
# arr = imgtoraw(img)
# g = open("C:/Users/user/Desktop/outlab1.txt", 'w', encoding='utf-8')
# for x in arr:
#     x = "".join([str(x) for x in RLEimg(x)])
#     g.write(x)
# g.close()
# img.close()



# f = open("C:/Users/user/Desktop/inlab.txt", 'r', encoding='utf-8')
# g = open("C:/Users/user/Desktop/outlab.txt", 'w', encoding='utf-8')
# n = f.readlines()
# for x in n:
#     x = RLE(x)
#     g.write(RLE(x))
# f.close()
# g.close()
