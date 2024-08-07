# zcalc
zhengxyz123's calculator. A simple scientific calculator in 1,000 lines!

## Usage
Type
```bash
python3 zcalc.py
```
in command line to enter interactive mode.

Or type
```bash
echo "1+2+3+4+5" | python3 zcalc.py
```
to calculate the expression immediately.

## Settings
You can use the following command to change calculator setttings:
```
set <a_setting_name>=<an_integer_value>
```

Available settings are:

| Name           | Description                                 | Default Value | Notes |
| :------------: | :-----------------------------------------: | :-----------: | :---: |
| precision      | Precision in display                        | 15            |       |
| base           | Numeric base in display (just for integers) | 10            | (1)   |
| enable_num2str | Whether or not to enable num2str            | 1             | (2)   |

Notes:

1. Available options are `2`, `8`, `10` and `16`.
2. Pass `0` to disable it. [Learn more about num2str](https://zhengxyz123.github.io/num2str).

There are some undocumented settings, you can find them inside the source code. Changing the values of these settings is not recommended.

To get the current value of a setting, enter:
```
get <a_setting_name>
```

## Readline Completer
If you can't remember the names of the settings described in the table above, as well as the names of all keywords and functions described below, zcalc provides the ability to use the tab key to complete the syntax.

This feature is not available on platforms that do not have the `readline` module installed.

## Calculating
zcalc only supports two types of numbers: `int` and `float`. Binary, octal and hexadecimal integers are also supported.

zcalc uses the same operator as Python, except:

| In zcalc | In Python |
| :------: | :-------: |
| `!`      | `~`       |
| `&&`     | `&`       |
| `\|\|`   | `\|`      |

zcalc provides a lot of functions, they are:

- Number-theoretic and representation functions: `abs(x)`, `ceil(x)`, `comb(n, k)`, `factorial(n)`, `floor(x)` and `perm(n, k=None)`
- Power and logarithmic functions: `cbrt(x)`(Python 3.11 and above), `exp(x)`, `expm1(x)`, `lg(x)`, `ln(x)`, `log(x, base=e)` and `sqrt(x)`
- Trigonometric functions: `acos(x)`, `asin(x)`, `atan(x)`, `atan(y, x)`, `cos(x)`, `sin(x)` and `tan(x)`
- Angular conversion functions: `degrees(x)` and `radians(x)`
- Hyperbolic functions: `acosh(x)`, `asinh(x)`, `atanh(x)`, `cosh(x)`, `sinh(x)` and `tanh(x)`
- Special functions: `erf(x)`, `erfc(x)`, `gamma(x)` and `lgamma(x)`

zcalc also defines 3 constants that cannot be reassigned: `e`, `pi` and `tau`.

The `ans` variable stores the result of the last calculation.

zcalc uses a five-point template to compute numerical differentiation. If you want to calculate $\frac{\mathrm{d}}{\mathrm{d}x}\sin x|_{x=\pi}$, type:
```
diff sin(x)|x=pi
```

zcalc uses Adaptive Simpson′s Rule to compute numerical differentiation, if you want to calculte $\int_1^2e^x\mathrm{d}x$, enter:
```
int e**x|x=1~2
```

zcalc uses Newton's method to solve equations. If you want to solve $\sin x=\frac{1}{2}$, type:
```
solve sin(x)-1/2|x=1
```

If you want to calculate $\sum_{n=1}^{100}2^n$, enter:
```
sum 2**n|x=1~100
```
