# string = raw_input()
string = 'abaacxbcbbbbacc cbc'
string = string.split(' ')
s1, s2 = string


def func(s1, s2):
    idx = []
    len2 = len(s2)
    if len2 == 1:
        if s2 not in s1:
            return -1, -1
        else:
            return 0
    else:
        for c in s2:
            if c not in s1:
                idx.append(-1)
                idx.append(-1)
                return idx[0], idx[-1]
            else:
                aa = s1.index(c)
                idx.append(aa)
                s1 = s1[(aa + 1):]
                s2 = s2[1:]

                temp = func(s1, s2)
                if type(temp) == type(1):
                    bb = []
                    bb.append(temp)
                    temp = bb
                temp = [i + aa + 1 for i in temp]
                return idx.extend(temp)


idx = func(s1, s2)
print idx[0], idx[-1]
