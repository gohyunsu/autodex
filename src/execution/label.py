"""Human-in-the-loop trial labeling prompt."""
from __future__ import annotations

import chime


def get_label():
    """Returns (success: bool|None, note: str|None).

    y       = success
    n       = fail
    c       = issue / skip (success=None)
    ym / nm = with memo
    """
    while True:
        chime.success()
        label = input("Label [y/n/c=issue / ym/nm=with memo]: ").strip().lower()
        if label == "y":
            return True, None
        if label == "ym":
            note = input("  Note: ").strip()
            return True, note or None
        if label == "n":
            return False, None
        if label == "nm":
            note = input("  Note: ").strip()
            return False, note or None
        if label == "c":
            note = input("  Note: ").strip()
            return None, note or "issue"
