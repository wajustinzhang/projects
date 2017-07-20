def twosum(list, target):
    visited = dict()
    for i in range(len(list)):
        if list[i] in visited:
            print(visited[list[i]], i)
        else:
            visited[target-list[i]] = i

    print('end')


def threesum(list, target):
    visited = {}
    for i in range(0,len(list) -1):
        sum = target-list[i];
        for j in range(1, len(list)):
            if sum-list[j] > 0:
                if list[j] in visited:
                    print(i, visited[list[j]], j)
                else:
                    visited[sum - list[j]] = j


if __name__ == '__main__':
    twosum([2, 3, 4, 1, 4, 6, 3, 8], 5)
    print('-----------')
    threesum([2, 3, 4, 1, 4, 6, 3, 8], 6)