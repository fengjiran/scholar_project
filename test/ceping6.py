s = raw_input()
a = s.split(' ')
n, m = [int(i) for i in a]

num = range(1, n + 1)
idx = []
for i in range(m):
    idx.append(int(raw_input()))

num1 = []
for i in range(m):
    num1.append(num[idx[i] - 1])

for i in range(m):
    num.remove(num1[i])
    num.insert(0, num1[i])

for i in range(n):
    print num[i]
