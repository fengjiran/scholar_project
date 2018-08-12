T = int(input())
salaries = []
for _ in range(T):
    a = int(input())
    salaries.append(a)

for a in salaries:
    if a <= 5000:
        print(0)
    elif a <= 8000:
        print(int(round((a - 5000) * 0.03)))
    elif a <= 17000:
        print(int(round(3000 * 0.03 + (a - 8000) * 0.1)))
    elif a <= 30000:
        print(int(round(3000 * 0.03 + (12000 - 3000) * 0.1 + (a - 17000) * 0.2)))
    elif a <= 40000:
        print(int(round(3000 * 0.03 + 9000 * 0.1 + 13000 * 0.2 + (a - 30000) * 0.25)))
    elif a <= 60000:
        print(int(round(3000 * 0.03 + 9000 * 0.1 + 13000 * 0.2 + 10000 * 0.25 + (a - 40000) * 0.3)))
    elif a <= 85000:
        print(int(round(3000 * 0.03 + 9000 * 0.1 + 13000 * 0.2 + 10000 * 0.25 + 20000 * 0.3 + (a - 60000) * 0.35)))
    else:
        print(int(round(3000 * 0.03 + 9000 * 0.1 + 13000 * 0.2 + 10000 * 0.25 + 20000 * 0.3 + 25000 * 0.35 + (a - 85000) * 0.45)))
