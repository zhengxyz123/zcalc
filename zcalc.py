import re
from typing import Iterator, NamedTuple


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
    keywords1 = ["exit", "get", "set", "solve", "diff", "int"]
    keywords2 = ["and", "or", "xor", "not"]
    operators = ["pow", "mul", "div", "plus", "minus", "range"]
    tokens = {
        "name": r"[A-Za-z_][A-Za-z0-9_]*",
        "pow": r"\*\*",
        "mul": r"\*",
        "div": r"/",
        "plus": r"\+",
        "minus": r"\-",
        "range": r"~",
        "lpar": r"\(",
        "rpar": r"\)",
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
        if kind == "name":
            if value in keywords1:
                kind = "keyword"
            elif value in keywords2:
                kind = "op"
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


if __name__ == "__main__":
    pass
