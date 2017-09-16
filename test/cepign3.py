string = raw_input()
length = len(string)

if length == 1:
    print length
elif length == 2:
    if string[0] == 'R' and string[1] == 'L':
        print 1
    else:
        print 2

    # for i in range(length - 1):
    #     c1 = string[i]
    #     c2 = string[i + 1]

    #     if c1 == 'R' and c2 == 'L':
    #         break
