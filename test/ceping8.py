num = int(raw_input())
x0, y0, z0 = [int(a) for a in raw_input().split(' ')]

result = 0
for i in range(num):
    mx, my, mz = [int(a) for a in raw_input().split(' ')]
    if (mx <= x0) and (my <= y0) and (mz <= z0):
        result += 1

print(result)
