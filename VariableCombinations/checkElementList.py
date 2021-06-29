

list1 = [1,2,3]
list2 = [2,2,2]

if (all(elem in list1  for elem in list2)):
    print("they are the same list")
else:
    print("they are not the same list")
