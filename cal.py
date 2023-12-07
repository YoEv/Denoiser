number = 687940 #34397
divisors = []

# 找出能整除 34397 的数字
for i in range(1, number + 1):
    if number % i == 0:
        divisors.append(i)

print("能整除 34397 的数字有：", divisors)
