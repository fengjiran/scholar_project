s = int(raw_input())
n = s / 3
if s % 3 == 0:
    n = int(n)
    print int('21' * n)
elif s % 3 == 1:
    n = int(n)
    print int('12' * n + '1')
elif s % 3 == 2:
    n = int(n)
    print int('21' * n + '2')
