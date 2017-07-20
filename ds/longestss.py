def longestsubstr(list, opernums):
    i=0
    j=1
    max = j-i
    repcharidx = 0
    size = len(list)

    while i<size-2 and j<size-1:
        if j - i > max:
            max = j-i
            repcharidx = j

        if list[j] == list[i]:
            j = j+1
        elif not found and j+1<size and list[j] != list[i] and list[j+1] == list[i]:
            j = j + 1
            found = True
        else:
            i=i+1
            j=j+1

    opr = opernums -1

    if i > opr:
        pass







