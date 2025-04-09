import math

def count(column):
    return len(column)

def sum(column):
    result = 0

    for v in column:
        result += v
    return result

def mean(column):
    return sum(column) / len(column)

def std(column):
    c_mean = mean(column)
    l_mean = len(column)

    c_sum = 0
    for v in column:
        c_sum = c_sum + (v - c_mean) ** 2
    return math.sqrt(c_sum / (l_mean - 1))

def max(column):
    r_max = -math.inf

    for v in column:
        r_max = v if r_max < v else r_max
    return r_max

def min(column):
    r_min = +math.inf

    for v in column:
        r_min = v if r_min > v else r_min
    return r_min

def percentile(column, percentage):
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

def sample_var(column):
    r_mean = mean(column)
    r_count = count(column)
    result = 0

    for v in column:
        result += (v - r_mean)**2
    return result / (r_count - 1)

def skew(column):
    r_count = count(column)
    r_std = std(column)
    r_mean = mean(column)

    result = 0
    for v in column:
        result += ((v - r_mean) / r_std) ** 3

    return (r_count / ((r_count - 1) * (r_count - 2))) * result
