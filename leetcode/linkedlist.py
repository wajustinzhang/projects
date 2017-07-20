class Node:
    def __init__(self, value, next):
        self.value = value
        self.next = next

    def __str__(self):
        return str(self.value)

def add(num1, num2, carryover = 0):
    if num2 == None:
        return

    while num1 != None:
        val = num1.value + num2.value + carryover
        if val < 10: 
            num1.value = val
            add(num1.next, num2.next)
        elif val >= 10:
            num1.value = 10 - val
            if num1.next == None and num2.next == None:
                num1.next = Node(1, None)
            elif num1.next == None and num2.next != None:
                num1.next = Node(num2.next.value + 1, None)
            elif num1.next != None and num2.next == None:
                add(num1.next, None, 1)
            else:
                add(num1.next, num2.next, 1)
