"""Formatação de números no padrão brasileiro."""


def thousands(value: int) -> str:
    """137256 → '137.256'."""
    return f"{value:,}".replace(",", ".")
