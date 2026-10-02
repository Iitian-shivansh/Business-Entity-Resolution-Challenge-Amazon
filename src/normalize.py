"""Text normalisation for business names and addresses.

Everything here is country-agnostic on purpose: the same rules run on US, India,
France and any other label that shows up. Country-specific knowledge is limited to
generic abbreviation tables (street types, legal suffixes, state names), which are
applied to every record regardless of its country label.
"""
import re

from anyascii import anyascii

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
_MOJIBAKE = re.compile(r"[âÂ]\x80[\x90-\x9f]")          # broken en-dash etc.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_VOWELS = re.compile(r"[aeiouy]")
_REPEAT = re.compile(r"(.)\1+")
_HARD_C = re.compile(r"c(?!h)")


_INDIC = re.compile(r"[\u0900-\u0dff]")
_ANUSVARA = re.compile(r"m(?=[kgcjtdns])")


def to_ascii_lower(s: str) -> str:
    """Transliterate any script (Devanagari, Tamil, accents...) to lowercase ASCII."""
    if not s:
        return ""
    s = _MOJIBAKE.sub("-", s)
    if s.isascii():
        return s.lower()
    indic = _INDIC.search(s) is not None
    s = anyascii(s).lower()
    if indic:
        # anyascii writes the nasal mark as 'm' ("marketimg", "imtrnesnl");
        # before a consonant it is almost always an 'n' in the English spelling.
        s = _ANUSVARA.sub("n", s)
    return s


def skeleton(tok: str) -> str:
    """Typo-tolerant phonetic key: first letter + consonants, repeats collapsed.

    farleigh / farlegh -> frlgh, green / grreen -> grn, foods / phuds -> fds
    """
    if not tok or tok.isdigit():
        return tok
    t = tok.replace("ph", "f").replace("ck", "k").replace("z", "s").replace("w", "v")
    # hard c / q sound like k ("consultants" vs transliterated "konsltents")
    t = _HARD_C.sub("k", t).replace("q", "k")
    t = _REPEAT.sub(r"\1", t)
    t = t[0] + _VOWELS.sub("", t[1:]) if len(t) > 1 else t
    t = t.replace("h", "") if len(t) > 2 else t
    return _REPEAT.sub(r"\1", t)


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
_NAME_NOISE = re.compile(
    r"\(\s*id\s*:?\s*\d+\s*\)"       # (ID: 43535)
    r"|&\s*#\s*\d+"                  # & #97816
    r"|#\d{3,}"                      # #12345
    r"|\.{2,}"                       # ...
)
_URL = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:co\.in|com|net|org|in|co|fr|biz|info|us|io)\b")

NAME_MAP = {
    "private": "pvt", "pvt": "pvt", "prvt": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    # transliterations of the same words written in Indic scripts
    "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivrr": "pvt",
    "praivat": "pvt", "pra": "pvt", "limitet": "ltd", "limirrd": "ltd", "li": "ltd",
    "limited": "ltd", "elelpi": "llp", "elelsi": "llc", "inteks": "inc",
    "tek": "tech", "eksports": "exports", "intrnesnl": "intl",
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc", "lnc": "inc",
    "company": "co", "co": "co", "comp": "co", "cos": "co",
    "international": "intl", "intl": "intl", "internationale": "intl",
    "services": "services", "service": "services", "svcs": "services", "svc": "services",
    "association": "assoc", "assoc": "assoc", "assn": "assoc", "asso": "assoc",
    "manufacturing": "mfg", "mfg": "mfg",
    "management": "mgmt", "mgmt": "mgmt",
    "group": "group", "grp": "group", "groupe": "group",
    "center": "center", "centre": "center", "ctr": "center",
    "technologies": "tech", "technology": "tech", "tech": "tech", "techno": "tech",
    "brothers": "bros", "bros": "bros",
    "department": "dept", "dept": "dept",
    "university": "univ", "univ": "univ",
    "enterprises": "ent", "enterprise": "ent", "ent": "ent",
    "and": "and", "et": "and", "und": "and",
    "compagnie": "cie", "cie": "cie",
    "societe": "societe", "soc": "societe",
    "etablissements": "ets", "ets": "ets", "etablissement": "ets",
    "saint": "st", "sainte": "st", "st": "st",
}

# Words that carry no identity (legal forms, articles). Removed for the "core" name.
NAME_STOP = {
    "pvt", "ltd", "corp", "inc", "co", "llc", "llp", "lp", "plc", "pllc", "pc", "pa",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "sca", "selarl", "scp", "scop",
    "gmbh", "ag", "bv", "nv", "dba", "aka", "the", "and", "of", "a", "an", "de", "du",
    "des", "la", "le", "les", "l", "d", "en", "dr", "mr", "ms", "mrs", "m", "s",
    "cie", "societe",
}


def _merge_single_letters(toks):
    """'l l c' -> 'llc', 's a r l' -> 'sarl'. Keeps other tokens intact."""
    out, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf))
    return out


_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def _fix_leet(t):
    """Typo digits inside words: 'po1yesters' -> 'polyesters', '5arl' -> 'sarl'.
    Only for mostly-alphabetic tokens, so '3m' or 'b2b' stay as they are."""
    if len(t) >= 4 and not t.isalpha() and not t.isdigit():
        if sum(ch.isalpha() for ch in t) >= len(t) - 2:
            return t.translate(_LEET)
    return t


def normalize_name(raw: str):
    """Return (full_norm, core_norm) strings for a business name."""
    s = to_ascii_lower(raw)
    s = _NAME_NOISE.sub(" ", s)
    s = s.replace("&", " and ").replace("'s ", "s ").replace("'", "")
    s = _URL.sub(r" \1 ", s)
    toks = [_fix_leet(t) for t in _NON_ALNUM.split(s) if t]
    toks = _merge_single_letters(toks)
    toks = [NAME_MAP.get(t, t) for t in toks]
    full = " ".join(toks)
    core = [t for t in toks if t not in NAME_STOP and not (t.isdigit() and len(t) >= 4)]
    if not core:                      # name made only of stop words: keep it all
        core = toks
    # collapse immediate duplicates ("better better", "fisheries fisheries")
    dedup = [t for i, t in enumerate(core) if i == 0 or t != core[i - 1]]
    return full, " ".join(dedup)


# --------------------------------------------------------------------------- #
# Addresses
# --------------------------------------------------------------------------- #
ADDR_MAP = {
    "street": "st", "st": "st", "str": "st", "strt": "st", "saint": "st", "sainte": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "aenue": "ave",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "crt": "ct",
    "circle": "cir", "cir": "cir",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd",
    "place": "pl", "pl": "pl", "plaza": "plz", "plz": "plz",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "terrace": "ter", "terr": "ter",
    "square": "sq", "sq": "sq",
    "suite": "ste", "ste": "ste",
    "floor": "fl", "fl": "fl", "flr": "fl",
    "apartment": "apt", "apt": "apt",
    "building": "bldg", "bldg": "bldg",
    "near": "near", "nr": "near",
    "opposite": "opp", "opp": "opp",
    "sector": "sector", "sec": "sector",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "rue": "rue", "r": "rue",
    "allee": "allee", "all": "allee", "alle": "allee",
    "impasse": "imp", "imp": "imp",
    "chemin": "chemin", "ch": "chemin",
    "route": "rte", "rte": "rte",
    "cours": "cours",
    "colony": "colony", "colny": "colony",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
}

ADDR_STOP = {
    "no", "number", "num", "ndeg", "door", "dor", "dotr", "plot", "flat", "house",
    "h", "hno", "null", "na", "none", "nil", "n/a", "unknown",
    "india", "usa", "us", "united", "france",
    "cdp", "borough", "township", "city", "village", "of",
    "de", "du", "des", "la", "le", "les", "l", "d", "the", "and", "c", "o",
}

_STATES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
    # India (English + common transliterations of native-script spellings)
    "uttar pradesh": "up", "uttr prdes": "up", "maharashtra": "mh", "mharastr": "mh",
    "delhi": "dl", "dilli": "dl", "karnataka": "ka", "krnatk": "ka",
    "tamil nadu": "tn", "tamilnadu": "tn", "tmilnatu": "tn", "gujarat": "gj", "gujrat": "gj",
    "west bengal": "wb", "pscimbng": "wb", "telangana": "tg", "telmgan": "tg", "ts": "tg",
    "rajasthan": "rj", "rajsthan": "rj", "nharastr": "mh", "kerln": "kl", "haryana": "hr", "hriyana": "hr",
    "madhya pradesh": "mp", "mdhy prdes": "mp", "bihar": "br", "kerala": "kl", "kerlm": "kl",
    "andhra pradesh": "ap", "amdhrprdes": "ap", "punjab": "pb", "pmjab": "pb",
    "odisha": "od", "orissa": "od", "od isa": "od", "jharkhand": "jh",
    "chhattisgarh": "cg", "assam": "as", "goa": "ga", "uttarakhand": "uk",
    "himachal pradesh": "hp", "jammu and kashmir": "jk", "chandigarh": "ch",
    # France (regions are long multi-word labels; map to short codes)
    "hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
    "ile de france": "idf", "auvergne rhone alpes": "ara", "occitanie": "occ",
    "grand est": "ges", "bretagne": "bre", "normandie": "nor",
    "provence alpes cote d azur": "pac", "bourgogne franche comte": "bfc",
    "centre val de loire": "cvl", "corse": "cor",
}
_STATE_PHRASES = sorted((k for k in _STATES if " " in k), key=len, reverse=True)
_STATE_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in _STATE_PHRASES) + r")\b")
_STATE_WORDS = {k: v for k, v in _STATES.items() if " " not in k}
STATE_CODES = set(_STATES.values())

_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_DIGIT_ALPHA = re.compile(r"(\d)([a-z])")
_ALPHA_DIGIT = re.compile(r"([a-z])(\d)")


def normalize_address(raw: str):
    """Return (addr_norm, numbers) where numbers is a space-joined sorted set."""
    s = to_ascii_lower(raw)
    s = s.replace("<null>", " ").replace("n/a", " ")
    s = _ORDINAL.sub(r"\1", s)
    s = _NON_ALNUM.sub(" ", s)
    s = _DIGIT_ALPHA.sub(r"\1 \2", s)
    s = _ALPHA_DIGIT.sub(r"\1 \2", s)
    s = _STATE_RE.sub(lambda m: _STATES[m.group(1)], s)
    toks = []
    for t in s.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = _STATE_WORDS.get(t, t)
            t = ADDR_MAP.get(t, t)
            if t in ADDR_STOP:
                continue
        toks.append(t)
    nums = sorted({t for t in toks if t.isdigit()})
    return " ".join(toks), " ".join(nums)


# Street-type / direction words: useful for similarity, useless as blocking keys.
ADDR_GENERIC = set(ADDR_MAP.values()) | STATE_CODES
