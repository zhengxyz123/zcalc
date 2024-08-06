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

zcalc uses a five-point template to compute numerical differentiation. If you want to calculate $\dfrac{\mathrm{d}}{\mathrm{d}x}\sin x|_{x=\pi}$, type:
```
diff sin(x)|x=pi
```

zcalc uses Adaptive Simpson′s Rule to compute numerical differentiation, if you want to calculte $\int\nolimits_1^2e^x\;\mathrm{d}x$, enter:
```
int e**x|x=1~2
```

zcalc uses Newton's method to solve equations. If you want to solve $\sin x=\dfrac{1}{2}$, type:
```
solve sin(x)-1/2|x=1
```

If you want to calculate $\sum\nolimits_{n=1}^{100}2^n$, enter:
```
sum 2**n|x=1~100
```
