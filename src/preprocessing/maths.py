import math

def sum(column: list):
    result = 0

    for v in column:
        result += v
    return result

def mean(column: list):
    return sum(column) / len(column)

def std(column: list):
    c_mean = mean(column)
    l_mean = len(column)

    c_sum = 0
    for v in column:
        c_sum = c_sum + (v - c_mean) ** 2
    return math.sqrt(c_sum / l_mean)

def max(column: list):
    r_max = -math.inf

    for v in column:
        r_max = v if r_max < v else r_max
    return r_max

def min(column: list):
    r_min = +math.inf

    for v in column:
        r_min = v if r_min > v else r_min
    return r_min

def percentile(column: list, percentage):
    percentage /= 100

    column.sort()

    n = len(column)
    i1 = math.floor((n - 1) * percentage)
    i2 = math.ceil((n - 1) * percentage)
    x1 = column[i1]
    x2 = column[i2]

    if i1 == i2:
        return x1
    else:
        return x1 + (percentage * (n - 1) - i1) * (x2 - x1)
