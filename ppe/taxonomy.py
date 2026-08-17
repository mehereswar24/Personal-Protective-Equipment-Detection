"""
Single source of truth for the PPE class taxonomy.

Replaces the three drifting copies the audit found (step1_prepare, step2_model,
run_pipeline each defined their own CLASSES / CLASS_TO_IDX / thresholds, with
different index bases and already-divergent values).

Design change (plan A1): we train POSITIVES ONLY and derive violations by
absence. There are no `no_*` object classes any more.

    helmet violation  = a `head` with no overlapping `helmet`
    vest/gloves/boots = a `person` track with no associated positive

Why: the old `no_helmet` class was 87% dataset4's `head` boxes (a whole-head
box convention) mixed with dataset1/5's "where a helmet would be" convention —
two different things trained as one class. And `no_gloves`/`no_boots` never
existed at all, which made glove/boot violations structurally impossible to
report. Detecting `head` as a first-class object fixes both.
"""

from __future__ import annotations

# ── Detector classes (background is index 0 in models that need it) ──
CLASSES: list[str] = ["person", "head", "helmet", "vest", "gloves", "boots"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Classes that are PPE items (i.e. can be "worn"). `person`/`head` are anatomy.
PPE_ITEMS: list[str] = ["helmet", "vest", "gloves", "boots"]

# Product-level classes we expose but do NOT currently detect reliably.
# `mask` is deliberately unsupported: its only source (dataset6) is a face-mask
# selfie dataset, which teaches "mask = large centred face" and does not
# transfer to CCTV. Kept here so the product can show it as beta/unsupported
# rather than silently reporting nonsense.
UNSUPPORTED: list[str] = ["mask"]


# ── Raw dataset label → our label ──
# Anything not listed maps to None and is DROPPED, but build_splits logs every
# unmapped label loudly (the old code swallowed them silently).
LABEL_MAP: dict[str, str | None] = {
    # person
    "person": "person", "Person": "person", "Human": "human_alias",
    # bare head (was: no_helmet)
    "head": "head", "no hat": "head", "no helmet": "head", "no-helmet": "head",
    # helmet
    "hat": "helmet", "helmet": "helmet", "Helmet": "helmet",
    # vest
    "vest": "vest", "Vest": "vest", "Safety Vest": "vest",
    # gloves
    "gloves": "gloves", "Gloves": "gloves", "Glove": "gloves",
    # boots
    "boots": "boots", "Boots": "boots", "Safety Boot": "boots",
    # explicitly dropped: negatives we no longer model as classes
    "no vest": None, "no-vest": None,
    "no gloves": None, "no boot": None, "no boots": None,
    # explicitly dropped: out of scope
    "mask": None, "Mask": None, "no-mask": None,
    "glasses": None, "Glass": None, "Ear-protection": None,
}
# `Human` is an alias of person; normalise after lookup so the mapping table
# stays readable.
_ALIASES = {"human_alias": "person"}


def map_label(raw: str) -> str | None:
    """Raw XML <name> → canonical class, or None if intentionally dropped.

    Raises KeyError for labels we have never seen, so new data fails loudly
    instead of being silently discarded (the old `.get(raw, "ignore")` hid
    typos and new classes alike).
    """
    label = LABEL_MAP[raw.strip()]
    return _ALIASES.get(label, label) if label else None


def is_known_label(raw: str) -> bool:
    return raw.strip() in LABEL_MAP


# ── Datasets ──
# dataset6 is excluded from the detector entirely (mask-only selfie data).
DATASET_DIRS: list[str] = [f"dataset{i}" for i in (1, 2, 3, 4, 5)]

# A class is only "supervised" in a dataset that actually annotates it.
# Everywhere else, an unannotated helmet/vest/glove is NOT a negative — it is
# unknown. build_splits derives this empirically (with MIN_ANNS_FOR_SUPERVISED)
# and writes it into the split file; the training loss then masks the
# unsupervised classes out per image. This is the fix for the audit's #2
# finding ("unlabeled negatives teach: no PPE = background").
MIN_ANNS_FOR_SUPERVISED = 50
