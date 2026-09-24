"""CLI gốc của project. Thay cho Makefile để chạy được trên Windows.

Mỗi subcommand import module của nó **bên trong** hàm, nên `python tasks.py
serve` không đòi phải có sẵn module eval, và ngược lại.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))


def cmd_ingest(rest: list[str]) -> int:
    from cli.ingest import main

    return main(rest)


def cmd_build(rest: list[str]) -> int:
    from cli.build_index import main

    return main(rest)


def cmd_querysets(rest: list[str]) -> int:
    from cli.make_querysets import main

    return main(rest)


def cmd_eval(rest: list[str]) -> int:
    from cli.evaluate import main

    return main(rest)


def cmd_serve(rest: list[str]) -> int:
    import uvicorn

    from app.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="tasks.py serve")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(rest)
    uvicorn.run(
        "app.main:get_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=args.reload,
    )
    return 0


COMMANDS = {
    "ingest": cmd_ingest,
    "build": cmd_build,
    "querysets": cmd_querysets,
    "eval": cmd_eval,
    "serve": cmd_serve,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(f"Dùng: python tasks.py <{' | '.join(COMMANDS)}> [tham số...]")
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
