"""A regra da arquitetura hexagonal: o domínio (regras e IAs) não depende da biblioteca gráfica."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "arena"
# Cenas, adaptadores e pontos de entrada podem usar pygame; o resto é domínio.
VIEW_MODULES = {
    "scene.py",
    "learning_scene.py",
    "final_scene.py",
    "board_view.py",
    "keyboard.py",
    "__main__.py",
    "record_all.py",
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_domain_modules_do_not_import_pygame_or_the_view():
    for path in PACKAGE.rglob("*.py"):
        if path.parent.name == "view" or path.name in VIEW_MODULES:
            continue
        imports = imported_modules(path)
        assert "pygame" not in imports, path
        assert not any(name.startswith("arena.view") for name in imports), path
