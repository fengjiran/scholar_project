import math
N = int(raw_input())
data = []
count_0 = 0
count_1 = 0
for _ in range(N):
    s = raw_input()
    s = s.split(',')
    s = [int(a) for a in s]
    if s[1]==0:
        count_0 += 1
    else:
        count_1 += 1
    data.append(s)
    
p0 = float(count_0)/N
p1 = float(count_1)/N
g = - p0 * math.log(p0, 2) - p1 * math.log(p1, 2)
print '%0.2f' % g