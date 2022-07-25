k =[5,6,6,1,3,2]
k_size= len(k)
k.sort

for i in range (k_size -2,-1,-1):
    if(k[i]!=k[k_size-1]):
        print("second largest is",k[i] )
        return
    else:
        print("There is no second largest input")

