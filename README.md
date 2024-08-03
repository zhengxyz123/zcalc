# zcalc
zhengxyz123's calculator. A simple scientific calculator in 1,000 lines!

## Usage
Type
```python
python3 zcalc.py
```
in command line to enter interactive mode.

Or type
```python
echo "1+2+3+4+5" | python3 zcalc.py
```
to calculate the expression immediately.

## Settings
You can use the following command to change calculator setttings:
```
set <a setting name> = <an integer value>
```

Available setting names are:

| Name           | Description                                 | Default Value | Notes |
| :------------: | :-----------------------------------------: | :-----------: | :---: |
| precision      | Precision in display                        | 15            |       |
| base           | Numeric base in display (just for integers) | 10            | (1)   |
| enable_num2str | Whether or not to enable num2str            | 1             | (2)   |

Notes:

1. Available options are `2`, `8`, `10` and `16`.
2. Pass `0` to disable it. [Learn more about num2str](https://zhengxyz123.github.io/num2str).
