#!/usr/bin/python3
# zcalc, a simple calculator.
#
# Copyright (c) 2024 zhengxyz123
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import re
import sys
from typing import Iterator, NamedTuple


__version__ = "0.9"
keywords = ["exit", "get", "set", "solve", "diff", "int"]


class ZCalcSyntaxError(Exception):
    def __init__(
        self, code: str, position: tuple[int, int], message: str | None = None
    ) -> None:
        self.code = code
        self.position = position
        self.message = message or f"syntax error at column {position[0] + 1}"
        super().__init__(self.message)


class Token(NamedTuple):
    type: str | None
    value: int | float | str
    where: tuple[int, int]


def tokenize(code: str) -> Iterator[Token]:
    operators = [
        "pow",
        "mul",
        "div",
        "mod",
        "plus",
        "minus",
        "lshift",
        "rshift",
        "and",
        "xor",
        "or",
        "not",
    ]
    tokens = {
        "name": r"[A-Za-z_][A-Za-z0-9_]*",
        "lpar": r"\(",
        "rpar": r"\)",
        "pow": r"\*\*",
        "mul": r"\*",
        "div": r"/",
        "mod": r"%",
        "plus": r"\+",
        "minus": r"\-",
        "lshift": r"<<",
        "rshift": r">>",
        "and": r"&&",
        "xor": r"\^",
        "or": r"\|\|",
        "not": r"!",
        "range": r"~",
        "equal": r"=",
        "comma": r",",
        "semi": r";",
        "sep": r"\|",
        "bnum": r"0b[01]+",
        "onum": r"0o[0-7]+",
        "hnum": r"0x[0-9a-fA-F]+",
        "dnum": r"[+\-]?\d+(\.\d*)?([Ee][+\-]?\d+)?",
        "skip": r"[ \t]+",
        "error": r".",
    }
    regex = "|".join(f"(?P<{name}>{text})" for name, text in tokens.items())
    for mo in re.finditer(regex, code):
        kind = mo.lastgroup
        value = mo.group()
        where = mo.start(), mo.end()
        if kind == "name" and value in keywords:
            kind = "keyword"
        elif kind in operators:
            kind = "op"
        elif kind == "bnum":
            kind = "num"
            value = int(value, base=2)
        elif kind == "onum":
            kind = "num"
            value = int(value, base=8)
        elif kind == "hnum":
            kind = "num"
            value = int(value, base=16)
        elif kind == "dnum":
            kind = "num"
            if "." in value or "e" in value.lower():
                value = float(value)
            else:
                value = int(value)
        elif kind == "skip":
            continue
        elif kind == "error":
            raise ZCalcSyntaxError(code, where)
        yield Token(kind, value, where)


def main() -> int:
    paraser = argparse.ArgumentParser(prog="zcalc", description="a simple calculator")
    paraser.add_argument(
        "-q", "--quiet", action="store_false", help="don't print initial banner"
    )
    paraser.add_argument(
        "-v", "--version", action="version", version=f"zcalc {__version__}"
    )
    args = paraser.parse_args()
    if not sys.stdin.isatty():
        exprs = (s.strip() for s in sys.stdin.readlines())
    if args.quiet:
        print(f"zcalc {__version__}, a simple calculator")
        print("Copyright (c) 2024 zhengxyz123")
        print("This is an open source software released under MIT license.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
