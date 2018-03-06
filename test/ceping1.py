string = raw_input()
length = len(string)
if length == 0:
    print length
else:
    stack = []
    for i in range(length):
        s = string[i]
        if s not in stack:
            stack.append(s)

        for j in range(i, length):
            c = string[j]
            if c == s:
                sub_str = string[i:j] + string[j]
            else:
                break

            if sub_str not in stack:
                stack.append(sub_str)

    print len(stack)
