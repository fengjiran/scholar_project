s = raw_input()
s1, s2 = s.split('-')
s1 = s1.split(',')
s1 = [int(a) for a in s1]

s2 = s2.split(':')
k = int(s2[-1])
s2 = s2[0].split(',')
s2 = [int(a) for a in s2]

print s1
print s2