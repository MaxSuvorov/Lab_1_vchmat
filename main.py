import numpy as np
import pandas as pd

# Вариант №22
a = 0.1
b = 0.6
x_star = 0.13
xi_minus_1 = 0.3
xi = 0.35
xi_plus_1 = 0.4


def f(x):
    return x ** 3 - np.cos(2 * x)


def second_d(x):
    return 6 * x + 4 * np.cos(2 * x)


def third_d(x):
    return 6 - 8 * np.sin(2 * x)


x_p = np.linspace(a, b, 11)
y_p = f(x_p)

table = y_p
df = pd.DataFrame(table, columns=['y'], index=x_p)
print('x', df)
print('\n')


def lag_first(x_i, x_i_plus_1, x_s):
    return f(x_i) * (x_s - x_i_plus_1) / (x_i - x_i_plus_1) + f(x_i_plus_1) * (x_s - x_i) / (x_i_plus_1 - x_i)


L1_x_star = lag_first(xi, xi_plus_1, x_star)
print("L1:", L1_x_star, '\n')


def R1(x, f_second_derivative, x_i, x_i_plus_1):
    interval = np.linspace(x_i, x_i_plus_1, 11)
    second_derivatives = f_second_derivative(interval)
    min_second_derivative = np.min(second_derivatives)
    max_second_derivative = np.max(second_derivatives)

    ω2 = (x - x_i) * (x - x_i_plus_1)

    R1_min = min_second_derivative * ω2 / 2
    R1_max = max_second_derivative * ω2 / 2

    return R1_min, R1_max


R1_min_value, R1_max_value = R1(x_star, second_d, xi, xi_plus_1)
print('min R1:', R1_min_value, '\n')
print('max R1:', R1_max_value, '\n')

R1_value = L1_x_star - f(x_star)
ans = R1_min_value < R1_value < R1_max_value
print('R1(x*):', R1_value, '\n')
print('Проверка равенства min R1 < R1(x*) < max R1:', ans, '\n')


def lag_second(x_i_minus_1, x_i, x_i_plus_1, x_s):
    form = [
        f(x_i_minus_1) * (x_s - x_i) / (x_i_minus_1 - xi) * (x_s - x_i_plus_1) / (x_i_minus_1 - x_i_plus_1),
        f(x_i) * (x_s - x_i_minus_1) / (x_i - x_i_minus_1) * (x_s - x_i_plus_1) / (x_i - x_i_plus_1),
        f(x_i_plus_1) * (x_s - x_i_minus_1) / (x_i_plus_1 - x_i_minus_1) * (x_s - x_i) / (x_i_plus_1 - x_i)
    ]
    return sum(form)


L2_x_star = lag_second(xi_minus_1, xi, xi_plus_1, x_star)
print('L2:', L2_x_star, '\n')


def R2(x, f_third_derivative, x_i_minus_1, x_i, x_i_plus_1):
    ω3 = (x - x_i_minus_1) * (x - x_i) * (x - x_i_plus_1)

    interval = np.linspace(x_i_minus_1, x_i_plus_1, 11)
    third_derivatives = f_third_derivative(interval)
    min_third_derivative = np.min(third_derivatives)
    max_third_derivative = np.max(third_derivatives)

    R2_min = min_third_derivative * ω3 / 6
    R2_max = max_third_derivative * ω3 / 6

    return R2_min, R2_max


R2_min_value, R2_max_value = R2(x_star, third_d, xi_minus_1, xi, xi_plus_1)
print('min R2:', R2_min_value, '\n')
print('max R2', R2_max_value, '\n')

# Проверка неравенства для остаточного члена
R2_value = L2_x_star - f(x_star)
ans = R2_min_value < R2_value < R2_max_value
print('R2(x*)', R2_value, '\n')
print('Проверка неравенства min R2 < R2(x*) < max R2:', ans, '\n')

# Этап 4
d1_f = (f(xi) - f(xi_minus_1)) / (xi - xi_minus_1)
d2_f = (f(xi_plus_1) - f(xi)) / (xi_plus_1 - xi)
d3_f = (d2_f - d1_f) / (xi_plus_1 - xi_minus_1)

data = {'xi': [xi_minus_1, xi, xi_plus_1],
        'd_f': ['-', d1_f, d2_f],
        'd2_f': ['-', '-', d3_f]}
df = pd.DataFrame(data)
print(df, '\n')

# Интерполяционные многочлены лагранжа
N1 = f(xi) + d2_f * (x_star - xi)
N2 = f(xi_minus_1) + d1_f * (x_star - xi_minus_1) + d3_f * (x_star - xi_minus_1) * (x_star - xi)
print('N1:', N1, '\n')
print('N2:', N2, '\n')

# Сравнение
diff1 = abs(L1_x_star - N1)
diff2 = abs(L2_x_star - N2)
print('L1 - L1(x*) =', diff1, '\n')
print('L2 - L2(x*) =', diff2, '\n')
