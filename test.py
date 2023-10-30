import time 
'''
get maximum from streaming data with constaints that it is logged within 60s of the time when the function is called.
'''
n_sample = 10

def get_samples(n_sample):
    data = []
    for i in range(n_sample):
        
        val = input('enter float: ')
        data.append((float(val), time.time()))
    print(data)
    return data
def get_max_in_1min(data):
    '''
        inputs:
            data: list[Tuple(val, time)] 
        output
            max_val: floats
    '''
    tmp = [data[0]]
    
    res = []
    for data_ in data: 
        if data_[1] - tmp[0][1] < 60:
            tmp.append(data_)
        else:
            tmp = tmp[1:]
            tmp.append(data_)
        res.append(max(tmp, key=lambda x: x[0]))
    return res

# samples = get_samples(n_sample)
# samples =[(0.9, 1698164818.7446568), 
#         (3.4, 1698164832.485812), 
#         (1.2, 1698164835.294216), 
#         (3.4, 1698164857.452843), 
#         (6.7, 1698164939.1239429), 
#         (4.5, 1698164942.823655), 
#         (9.0, 1698164945.869289), 
#         (5.6, 1698164948.885738), 
#         (4.5, 1698164954.202194), 
#         (3.9, 1698164988.0416691)]
# print(get_max_in_1min(samples)) 
'''
Generator for even numbers
'''
def even_generator(limit=10):
    for i in range(2, limit, 2):
        yield i

# for num in even_generator(10):
#     print(num)

'''
n-th root of m
'''
def find_n_th_root(x, n):
    init = 1
    eps = 0.0000001
    while abs(x - init**n) > eps:
        init = init + (x - init**n) / (n * init**(n-1))
    return init

# print(find_n_th_root(10, 3))

