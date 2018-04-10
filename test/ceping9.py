def solution(num):
    def step(num):
        num_eles = []
        while num >= 10:
            i = num % 10
            num = num / 10
            num_eles.append(i)
        num_eles.append(num)
        num = sum([a**2 for a in num_eles])
        if num >= 10:
            num = step(num)
        return num
    num = step(num)

    if num == 1:
        return 'true'
    else:
        return 'false'


if __name__ == '__main__':
    _num = int(raw_input())
    res = solution(_num)
    print res + '\n'
