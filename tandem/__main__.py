"""
tandem.__main__
---------------
Minimal CLI entrypoint for ``python -m tandem``.

Provides ``--health-check`` and verbose logging flags so the package can be
exercised in CI without writing a separate driver script.
"""

import argparse
import logging
import sys

from . import health_check


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tandem",
        description="tandem-solar package CLI",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run a self-test of core models and exit (0 = pass, 1 = fail)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v=INFO, -vv=DEBUG)",
    )
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.health_check:
        rc = health_check()
        if rc == 0:
            print("✓ Health check passed")
        else:
            print("✗ Health check FAILED", file=sys.stderr)
        sys.exit(rc)

    parser.print_help()


if __name__ == "__main__":
    main()
