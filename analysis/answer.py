from __future__ import annotations

import json

try:  # Support running as script or module
    from .service import generate_analysis, load_mock_request
except ImportError:  # pragma: no cover
    from service import generate_analysis, load_mock_request  # type: ignore


def main() -> None:
    result = generate_analysis(load_mock_request())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()