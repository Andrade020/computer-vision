"""
Typographic metrics for handwriting layout.

Baseline = 0.0. Units are relative to the nominal x-height (=1.0).
Each character maps to a vertical box (bottom, top) that the glyph
should occupy, so ascenders/descenders/caps land in natural positions.
"""

# Vertical extents (bottom, top) relative to baseline, x-height = 1.0
X_TOP = 1.0
ASC_TOP = 1.42       # ascenders / caps / digits
DESC_BOT = -0.42     # descenders
CAP_TOP = 1.40

_DESCENDERS = set("gjpqy")
_ASCENDERS = set("bdfhklt")
_XHEIGHT = set("aceimnorsuvwxz")

# characters whose lowercase form is essentially the uppercase shape scaled
CASE_SAME = set("ckmopsuvwxz")


def char_box(ch):
    """Return (bottom, top) vertical extent for a single character."""
    if ch.isspace():
        return (0.0, X_TOP)
    if ch.isdigit():
        return (0.0, ASC_TOP)
    if ch.isupper():
        return (0.0, CAP_TOP)
    if ch in _DESCENDERS:
        return (DESC_BOT, X_TOP)
    if ch in _ASCENDERS:
        return (0.0, ASC_TOP)
    if ch in _XHEIGHT:
        return (0.0, X_TOP)
    # punctuation / symbols
    if ch == ",":
        return (-0.18, 0.22)
    if ch in "._":
        return (0.0, 0.28)
    if ch in "'\"`^":
        return (X_TOP, ASC_TOP)
    if ch in "-=~+*":
        return (0.45, 0.75)
    if ch in "()[]{}|/\\":
        return (DESC_BOT, ASC_TOP)
    if ch in ":;":
        return (0.0, X_TOP)
    # default: x-height box
    return (0.0, X_TOP)


# Nominal advance width (in x-height units) used when a glyph is missing
# or as a spacing hint. Real width comes from the glyph aspect ratio.
DEFAULT_ADVANCE = 0.62
SPACE_ADVANCE = 0.55
