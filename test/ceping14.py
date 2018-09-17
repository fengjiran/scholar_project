s1 = raw_input()
s2 = raw_input()

def lcs(s1, s2):
    lens1 = len(s1)
    lens2 = len(s2)
    
    s3 = [[0 for i in range(lens2+1)] for j in range(lens1+1)]
    flag = [[0 for i in range(lens2+1)] for j in range(lens1+1)]
    
    for i in range(lens1):
        for j in range(lens2):
            if s1[i] == s2[j]:
                s3[i+1][j+1] = s3[i][j] + 1
                flag[i+1][j+1] = 'ok'
            elif s3[i+1][j] > s3[i][j+1]:
                s3[i+1][j+1] = s3[i+1][j]
                flag[i+1][j+1] = 'left'
            else:
                s3[i+1][j+1] = s3[i][j+1]
                flag[i+1][j+1] = 'up'
    
    return flag
ss = []

def printlcs(flag, a, i, j):
    # cnt = []
    if i == 0 or j==0:
        return
    if flag[i][j] == 'ok':
        printlcs(flag, a, i-1, j-1)
        # print(a[i-1])
        ss.append(a[i-1])
    elif flag[i][j] == 'left':
        printlcs(flag, a, i, j-1)
    else:
        printlcs(flag, a, i-1, j)

    # return cnt

flag = lcs(s1, s2)
printlcs(flag, s1, len(s1), len(s2))
print(len(ss))