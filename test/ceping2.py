s1 = raw_input()
s2 = raw_input()
a1 = s1.split(' ')
a1 = [int(i) for i in a1]

a2 = s2.split(' ')
a2 = [int(i) for i in a2]

n, k = a1
sum_k = 0
results = []

for i in range(k):
    sum_k += a2[i]

results.append(sum_k)

tmp = a2[0:k]

len_2 = len(a2)

for i in range(k, len_2):
    min_ = min(tmp)
    if min_ < a2[i]:
        tmp.remove(min_)
        tmp.append(a2[i])

    sum_k = 0
    for _ in tmp:
        sum_k += _
    results.append(sum_k)

results = [str(s) for s in results]

print ' '.join(results)
