N = int(raw_input())
s = raw_input()
s = s.split(' ')
s = [int(a) for a in s]

sums = []
max_value = max(s)
min_value = min(s)
max_idx = s.index(max_value)
min_idx = s.index(min_value)

a = 0
b = 0
if min_idx < max_idx:
    a = min_idx
    b = max_idx
else:
    a = max_idx
    b = min_idx

dis = max_value - min_value
for i in range(N):
    for j in range(i, N):
        if (i>=a) and (j<=b):
            sums.append(dis)
        else:
            temp = s[i:j+1]
            temp_min = min(temp)
            temp_max = max(temp)
            sums.append(temp_max-temp_min)
        

print sum(sums)