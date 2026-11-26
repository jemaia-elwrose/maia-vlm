"""maia-vlm: check detector events with a vision-language model.

from maia_vlm import Backend, Event, find_check, verify_clip

verdict = verify_clip("clip.mp4", Event(label="person", box=(410, 120, 520, 400), at_s=3.2),
                      find_check("helmet"), Backend.from_env())
print(verdict.verdict, verdict.description)
"""

from .backends import Backend, BackendError, Reply
from .checks import Check
from .checks import find as find_check
from .checks import load_all as load_checks
from .frames import Event, Frame, read_clip
from .verify import Verdict, verify, verify_clip

__version__ = "0.1.1"

__all__ = [
    "Backend",
    "BackendError",
    "Check",
    "Event",
    "Frame",
    "Reply",
    "Verdict",
    "find_check",
    "load_checks",
    "read_clip",
    "verify",
    "verify_clip",
]
