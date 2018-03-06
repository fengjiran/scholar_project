import heapq
import copy
K = raw_input()
N = raw_input()

K = int(K)
N = int(N)

arr = []
num = []
idx = []

for _ in range(K):
    a = raw_input()
    a = a.split(' ')
    a = [int(i) for i in a]
    a.sort()
    arr.append(a[0])
    # idx.append(0)
    num.append(a)


idx = arr.index(min(arr))

heapq.heapify(arr)
minVal = arr[0]
maxVal = arr[-1]

while 1:
    temp = copy.copy(arr)
    heapq.heappop(arr)
    heapq.heappush(arr, num[idx][i])
    minVal = arr[0]
    maxVal = arr[-1]

    idx = temp.index(arr[0])


print minVal, maxVal
