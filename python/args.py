import sys

# read args
print(sys.argv)

# string format
print('test {}'.format('python'))

# string join
print('test'.join('-----'))

# reverse a string
'hello world'[::-1]

oldList = [1,3,4]
# shallow copy
newList = oldList[:]

def foo(x='thisis'):
    pass

def foo(x, *y, **z):
    pass

def foo(x, y='test'):
    pass

def foo(x, l=[]):
    pass

def concat(*args, sep="/"):
    return sep.join(args)

def lbd(n):
    return lambda x:x**n

# List, tuple, set, dict, sequence

squares = list(map(lambda x: x**2, range(10)))

