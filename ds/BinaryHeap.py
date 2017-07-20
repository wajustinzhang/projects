# Max binary heap
class MaxBH:
    def __index__(self, capacity):
        self.pg = [0 for i in capacity]
        self.size = 0

    def isEmpty(self):
        return self.size == 0

    def insert(self, key):
        if self.size < len(self.pg):
            return

        self.size = self.size + 1
        self.pg[self.size-1] = key

    def delMax(self):
        max = self.pg[0]
        self.exchange(0, self.size-1)
        self.size = self.size -1
        self.sink(1)
        self.pg[self.size-1] = None
        return max

    def __swim(self, k):
        while k>1 and self.less(k/2, k):
            self.exchange(k, k/2)
            k=k/2

    def __sink(self, k):
        while 2*k < self.size:
            j = 2*k
            if j<self.size and self.less(j, j+1):
                j = j+1
            if self.less(k,j):
                break

            self.exchange(k, j)

    def less(self, i, j):
        return self.pg[i] < self.pg[j]

    def exchange(self, i, j):
        tmp = self.pg[i]
        self.pg[i] = self.pg[j]
        self.pg[j] = tmp


