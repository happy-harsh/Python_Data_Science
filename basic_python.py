# x =5.6
# m=6
# y ="9"
# z="2"
#
# print(type(x))
# print(type(y))
# print(x+m)
# print(y+z)



#float to string
# a = str(5.6)
# print(a)
#
# c = "haappy"
# print(a+c)

#string to float
# b = float("hi")
# print(b)


'''
hello world

'''

# lists  ordered , changeable(mutuable), allows duplicate
# mylist = [98,str(5.6),float(3),'harsh',['happy',5.6,'']]
#
# l2=mylist[4][1]
# print(l2)
#
# mylist[4][1]="harsh"#2nd elment at index 2
# print(mylist)
#
# mylist2 = [98,str(5.6),float(3),'harsh',['happy',5.6,'']]
#
# print(mylist2[1:4]) #it will print elements from index 1 to index 6-1=5
# print(mylist2[-5:-4])

# TUPLE  ordered , unchangeable(immutuable), allows duplicates
# mytup=('hello',str(5.6),45,23,45)
# t = mytup[1][0]
# print(t)
# # mytup[0][0]="j"  no item assignment possible
# print(mytup[0::2])  #internal is done on index ie for 2 internal element with index 0 2 4 5 will be printed


# dictionary  unordered , changeable(mutuable), no allows duplicate,slicing not possible

mydict = {12:["pqrst",3,4],3:'bd',5:'c',7:'d'}
print(mydict[12][0][0]) #mydict[key][indices of value]


# mydict[3][0]="happy"   *string does not support item assignment whereas list does*
# print(mydict)

# slicing not possible
# print(mydict[:])