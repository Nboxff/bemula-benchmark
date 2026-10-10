from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from bemula.dialogues.generator import _BASE_DATE
from bemula.dialogues.planner import is_clean_value
from bemula.llm import BaseLLM
from bemula.schemas import AtomicFact, FactCategory, Sensitivity, UserProfile

# Every event fact must predate the instance timeline (which starts at
# _BASE_DATE), otherwise the user simulator's temporal-consistency rules have
# no stable reference and the profile can contradict the conversation's "now".
_REFERENCE_MONTH = _BASE_DATE.strftime("%Y-%m")

# Seed attribute option lists; each profile samples one value per attribute
# independently (see `_sample_seed_attrs`). Age and the region- and
# gender-matched name are sampled per profile too: identity attributes are
# fixed seeds, never LLM-generated atomic facts.
SEED_ATTRIBUTES: dict[str, list[str]] = {
    "gender": ["male", "female"],
    "occupation": [
        "software engineer",
        "nurse",
        "teacher",
        "accountant",
        "product manager",
        "graphic designer",
        "data scientist",
        "lawyer",
        "chef",
        "dentist",
        "journalist",
        "electrician",
    ],
    "region": [
        "Seattle", "Austin", "Chicago", "Denver", "Boston", "San Diego",
        "New York", "San Francisco", "Miami", "Beijing", "Shanghai",
        "Tokyo", "London", "Paris", "Singapore", "Berlin", "Seoul",
        "Madrid", "Sydney", "Toronto",
    ],
    "family_structure": ["single", "married", "married with kids", "living with parents"],
}

# Persona names, localized per region and matched to the persona's gender.
# Each profile picks a name deterministically from its region's pool for its
# gender, so a Beijing persona gets a Chinese name, a Berlin persona a German
# one, and a "male" persona never gets a female-coded name. Pools are kept
# as large as practical (English 75/83, Chinese 36/36, Japanese 24/24,
# French 24/24, German 28/28, Korean 24/24, Spanish 24/24) so the formal
# scale (50 profiles) keeps same-name reuse low.
_NAME_POOLS: dict[str, dict[str, list[str]]] = {
    "en": {
        "male": [
            "Adam", "Alan", "Albert", "Alex", "Alexander", "Andrew",
            "Anthony", "Arthur", "Benjamin", "Blake", "Brandon", "Brian",
            "Carl", "Charles", "Chris", "Christopher", "Connor", "Daniel",
            "David", "Dennis", "Derek", "Donald", "Douglas", "Edward",
            "Eric", "Ethan", "Evan", "Frank", "George", "Gerald", "Gordon",
            "Gregory", "Henry", "Ian", "Isaac", "Jack", "Jacob", "James",
            "Jason", "Jeffrey", "Jeremy", "John", "Jonathan", "Joseph",
            "Joshua", "Justin", "Kevin", "Kyle", "Liam", "Louis", "Luke",
            "Mark", "Martin", "Matthew", "Michael", "Nathan", "Nicholas",
            "Noah", "Oliver", "Patrick", "Paul", "Peter", "Philip",
            "Raymond", "Richard", "Robert", "Roger", "Ryan", "Samuel",
            "Scott", "Sean", "Steven", "Thomas", "Victor", "William",
        ],
        "female": [
            "Abigail", "Alice", "Amanda", "Amelia", "Amy", "Andrea",
            "Angela", "Anna", "Anne", "Ashley", "Audrey", "Barbara",
            "Beatrice", "Bella", "Betty", "Brenda", "Brittany", "Brooke",
            "Catherine", "Charlotte", "Chloe", "Claire", "Clara", "Cynthia",
            "Diana", "Dorothy", "Eleanor", "Elizabeth", "Ella", "Emily",
            "Emma", "Erin", "Eve", "Fiona", "Grace", "Hannah", "Heather",
            "Helen", "Irene", "Isabella", "Ivy", "Jasmine", "Jennifer",
            "Jessica", "Julia", "Karen", "Katherine", "Kelly", "Kimberly",
            "Laura", "Lauren", "Leah", "Lily", "Linda", "Lisa", "Lucy",
            "Lydia", "Margaret", "Maria", "Mary", "Megan", "Melissa",
            "Mia", "Michelle", "Molly", "Nancy", "Nicole", "Olivia",
            "Rachel", "Rebecca", "Rose", "Ruth", "Samantha", "Sandra",
            "Sarah", "Sophia", "Stella", "Susan", "Tessa", "Victoria",
            "Wendy", "Yvonne", "Zoe",
        ],
    },
    "zh": {
        "male": [
            "Wei Li", "Qiang Wang", "Lei Zhang", "Jun Liu", "Yong Chen",
            "Tao Yang", "Ming Zhao", "Jie Huang", "Bin Zhou", "Peng Wu",
            "Kai Xu", "Chao Sun", "Yang Hu", "Hao Guo", "Long He", "Fei Lin",
            "Gang Luo", "Bo Zheng", "Jian Ma", "Feng Zhu", "Yi Gao",
            "Hui Tang", "Wei Han", "Jun Cao", "Lei Xu", "Ming Song",
            "Peng Deng", "Chao Xie", "Yang Feng", "Hao Lu", "Bin He",
            "Kai Lin", "Jie Zhou", "Yong Zhang", "Qiang Liu", "Feng Wang",
        ],
        "female": [
            "Fang Li", "Na Wang", "Min Zhang", "Jing Liu", "Ting Chen",
            "Yan Yang", "Xue Zhao", "Lin Huang", "Juan Zhou", "Xia Wu",
            "Mei Sun", "Hui Xu", "Qian Hu", "Ying Guo", "Ling He", "Xin Luo",
            "Dan Zheng", "Wen Ma", "Li Zhu", "Jia Gao", "Lei Deng",
            "Sha Han", "Yun Cao", "Fei Xu", "Man Song", "Shuang Tang",
            "Rui Feng", "Wan Lu", "Jie Xie", "Qin Chen", "Hong Yang",
            "Ping Zhao", "Yan Wang", "Jing Chen", "Min Li", "Ting Zhang",
        ],
    },
    "ja": {
        "male": [
            "Haruto Sato", "Yuto Tanaka", "Sota Suzuki", "Riku Takahashi",
            "Kaito Watanabe", "Ren Yamamoto", "Daiki Kobayashi",
            "Ryo Nakamura", "Shota Ito", "Taiga Kato", "Ryota Yoshida",
            "Kento Yamada", "Takumi Sasaki", "Hiroshi Matsumoto",
            "Kenji Inoue", "Kazuki Kimura", "Sho Hayashi",
            "Tsubasa Yamazaki", "Akihiro Mori", "Daisuke Ogawa",
            "Tetsuya Fujita", "Keita Saito", "Naoki Ota", "Takahiro Ishikawa",
        ],
        "female": [
            "Yui Tanaka", "Aoi Suzuki", "Hana Takahashi", "Sakura Watanabe",
            "Mio Kato", "Rina Yamamoto", "Yuna Kobayashi", "Mei Nakamura",
            "Nana Ito", "Akari Yoshida", "Hina Yamada", "Kanna Sasaki",
            "Emi Matsumoto", "Aya Inoue", "Misaki Kimura", "Haruka Hayashi",
            "Chihiro Yamazaki", "Rin Mori", "Yuki Saito", "Kaede Fujita",
            "Saki Ogawa", "Momoka Ota", "Ayumi Ishikawa", "Natsumi Ueda",
        ],
    },
    "fr": {
        "male": [
            "Lucas Martin", "Hugo Bernard", "Louis Dubois", "Antoine Girard",
            "Jules Fournier", "Thomas Moreau", "Nathan Laurent",
            "Enzo Lefebvre", "Gabriel Roux", "Adam Fontaine", "Leo Mercier",
            "Raphael Blanc", "Maxime Garnier", "Arthur Chevalier",
            "Theo Rousseau", "Paul Lambert", "Nicolas Michel", "Quentin David",
            "Baptiste Petit", "Clement Renard", "Romain Robert",
            "Julien Vincent", "Mathis Simon", "Alexandre Perrin",
        ],
        "female": [
            "Camille Laurent", "Emma Dubois", "Chloe Petit", "Lea Martin",
            "Manon Lefevre", "Zoe Bernard", "Louise Moreau", "Jade Girard",
            "Lena Fournier", "Julia Roux", "Sarah Fontaine", "Clara Mercier",
            "Alice Blanc", "Charlotte Garnier", "Eva Chevalier",
            "Ines Rousseau", "Juliette Lambert", "Nina Michel",
            "Margaux David", "Lucie Vincent", "Anna Simon", "Laura Perrin",
            "Romane Renard", "Pauline Robert",
        ],
    },
    "de": {
        "male": [
            "Lukas Weber", "Jonas Müller", "Felix Schmidt", "Maximilian Fischer",
            "Leon Wagner", "Paul Becker", "Finn Hoffmann", "Erik Schäfer",
            "Tobias Koch", "Stefan Bauer", "Julian Richter", "Niklas Klein",
            "Adrian Wolf", "David Schröder", "Martin Neumann", "Simon Schwarz",
            "Fabian Zimmermann", "Patrick Braun", "Daniel Krüger",
            "Moritz Hofmann", "Tim Hartmann", "Florian Lange",
            "Alexander Schmitt", "Benjamin Werner", "Jan Krause", "Nils Meier",
            "Markus Lehmann", "Thomas Huber",
        ],
        "female": [
            "Anna Weber", "Lena Müller", "Julia Schmidt", "Laura Fischer",
            "Sophie Wagner", "Marie Becker", "Hannah Hoffmann", "Emma Schäfer",
            "Mia Koch", "Johanna Bauer", "Katharina Richter", "Nina Klein",
            "Lisa Wolf", "Sarah Schröder", "Clara Neumann", "Lea Schwarz",
            "Amelie Zimmermann", "Paula Braun", "Nele Krüger",
            "Frieda Hofmann", "Greta Hartmann", "Ines Lange", "Theresa Schmitt",
            "Elisa Werner", "Miriam Krause", "Carolin Meier", "Vanessa Lehmann",
            "Jule Huber",
        ],
    },
    "ko": {
        "male": [
            "Min-jun Kim", "Seo-jun Lee", "Do-yun Park", "Ha-joon Choi",
            "Ji-ho Jung", "Jun-seo Kang", "Eun-woo Cho", "Hyun-woo Yoon",
            "Tae-yang Jang", "Woo-jin Im", "Joon-ho Han", "Sung-min Oh",
            "Dong-hyun Seo", "Jin-woo Shin", "Sang-hoon Kwon",
            "Min-seok Hwang", "Yong-jun Ahn", "Ki-hoon Song", "Ji-min Bae",
            "Hyeon-woo Hong", "Seung-min Yoo", "Dong-wook Moon",
            "Tae-hyun Baek", "Jae-won Nam",
        ],
        "female": [
            "Seo-yeon Kim", "Min-seo Lee", "Ji-woo Park", "Ha-eun Choi",
            "Soo-ah Jung", "Ye-jin Kang", "Yu-na Cho", "Eun-ji Yoon",
            "Hye-jin Jang", "So-yeon Im", "Ji-ae Han", "Na-yeon Oh",
            "Da-eun Seo", "Su-jin Shin", "Mi-so Kwon", "Yeon-woo Hwang",
            "Ga-eun Ahn", "Bo-ram Song", "Hye-won Bae", "Ji-yoo Hong",
            "Ah-reum Yoo", "Chae-won Moon", "Ha-yoon Baek", "Seo-hyun Nam",
        ],
    },
    "es": {
        "male": [
            "Alejandro García", "Javier Fernández", "Diego González",
            "Carlos Rodríguez", "Miguel López", "Fernando Martínez",
            "Rafael Sánchez", "Andrés Pérez", "Pablo Gómez", "Sergio Martín",
            "Jorge Jiménez", "Daniel Ruiz", "David Hernández", "Marcos Díaz",
            "Álvaro Moreno", "Adrián Muñoz", "Hugo Álvarez", "Martín Romero",
            "Mario Navarro", "Iván Torres", "Rubén Domínguez", "Óscar Vázquez",
            "Gonzalo Ramos", "Enrique Gil",
        ],
        "female": [
            "María García", "Lucía Fernández", "Carmen González",
            "Ana Rodríguez", "Laura López", "Elena Martínez", "Sofía Sánchez",
            "Paula Pérez", "Marta Gómez", "Alba Martín", "Sara Jiménez",
            "Claudia Ruiz", "Andrea Hernández", "Nerea Díaz", "Celia Moreno",
            "Irene Muñoz", "Cristina Álvarez", "Patricia Romero",
            "Rocío Navarro", "Silvia Torres", "Beatriz Domínguez",
            "Raquel Vázquez", "Julia Ramos", "Alicia Gil",
        ],
    },
}

# region seed -> name-pool key. English-speaking regions (incl. Singapore,
# Sydney and Toronto) share the English pool; the other regions get localized
# pools.
_REGION_NAME_POOL: dict[str, str] = {
    "Seattle": "en", "Austin": "en", "Chicago": "en", "Denver": "en",
    "Boston": "en", "San Diego": "en", "New York": "en", "San Francisco": "en",
    "Miami": "en", "London": "en", "Singapore": "en", "Sydney": "en",
    "Toronto": "en",
    "Beijing": "zh", "Shanghai": "zh",
    "Tokyo": "ja",
    "Paris": "fr",
    "Berlin": "de",
    "Seoul": "ko",
    "Madrid": "es",
}


def _sample_seed_attrs(rng: random.Random) -> dict[str, str]:
    """Randomly sample one complete seed-attribute set for one profile.

    Identity attributes come first so the name is drawn from the region- and
    gender-matched pool; age is uniform in 18-65.
    """
    region = rng.choice(SEED_ATTRIBUTES["region"])
    gender = rng.choice(SEED_ATTRIBUTES["gender"])
    pool_key = _REGION_NAME_POOL.get(region, "en")
    pool = _NAME_POOLS[pool_key].get(gender, _NAME_POOLS[pool_key]["female"])
    return {
        "gender": gender,
        "occupation": rng.choice(SEED_ATTRIBUTES["occupation"]),
        "region": region,
        "family_structure": rng.choice(SEED_ATTRIBUTES["family_structure"]),
        "age": str(rng.randint(18, 65)),
        "name": rng.choice(pool),
    }


def generate_profiles(
    llm: BaseLLM,
    config: dict[str, Any],
    limit: int | None = None,
    start_index: int = 0,
) -> list[UserProfile]:
    """Generate user profiles per config scale (LLM-only).

    Profiles are synthesized concurrently (one LLM call each). Failure policy:
    each profile is retried up to
    `generation.max_retries` times and then skipped with a warning — a
    profile is never emitted in a degraded form.

    `start_index` shifts profile ids to `profile-{start_index + i}` and
    advances the shared seed-attributes RNG past the skipped indices, so a
    top-up run yields fresh personas whose ids never collide with an earlier
    batch.
    """
    num = config.get("scale", {}).get("num_profiles", 100)
    if limit is not None:
        num = min(num, limit)
    rng = random.Random(config.get("seed", 42))
    # Per-profile seed attributes, sampled up front so the shared RNG is only
    # touched from one thread (deterministic under a fixed seed).
    seeds = [_sample_seed_attrs(rng) for _ in range(start_index + num)]
    seeds = seeds[start_index:]
    workers = int(config.get("generation", {}).get("max_workers", 1))

    def build(i: int) -> tuple[int, UserProfile | None]:
        seed_attrs = seeds[i - start_index]
        attempts = config.get("generation", {}).get("max_retries", 3)
        for attempt in range(attempts):
            try:
                return i, _llm_profile(i, seed_attrs, llm, config)
            except Exception as e:
                if attempt == attempts - 1:
                    print(f"  warning: profile {i} failed after {attempts} attempts "
                          f"({e}); skipping")
                    return i, None
                print(f"  profile {i} attempt {attempt + 1} failed ({e}); retrying")
        raise AssertionError("unreachable")

    profiles: list[UserProfile | None] = [None] * num
    indices = range(start_index, start_index + num)
    with tqdm(total=num, desc="profiles", unit="profile") as bar:
        if workers <= 1:
            for i in indices:
                i, profile = build(i)
                profiles[i - start_index] = profile
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(build, i) for i in indices]
                for fut in as_completed(futures):
                    i, profile = fut.result()
                    profiles[i - start_index] = profile
                    bar.update(1)
    return [p for p in profiles if p is not None]


_FACTS_PROMPT = """You are building a synthetic user profile for a benchmark.

Persona seed attributes: {seed_attrs}

The persona's age, gender and name are fixed seed attributes given above; \
do NOT emit atomic facts about them (no facts stating the persona's age, \
gender, or name) — those live in the demographics, not in the fact list.

The present date is {present_month}; the user will chat with the assistant \
from this date on. Every event you generate MUST be in the past relative to \
it: "event_time" must be strictly before {ref_month}, and dates embedded in \
values must agree with that.

Generate {n_facts} atomic facts about this user. Each element of the "facts" array:
{{"category": one of address/health/preference/career/education/relationship/finance/routine/event/hobby/social/other,
  "subject": "{name}", "predicate": short relation phrase, "value": the fact value,
  "surface_form": a first-person sentence stating the fact,
  "event_time": "YYYY-MM" or null, "sensitivity": one of low/medium/high}}

Cover ALL 12 categories, and make the profile feel like a real life: include
several facts in each of hobby (leisure pursuits) and social (friends, clubs,
community ties), and a RICH set of "what did the user do" facts.

Also include at least 8 TIME-ANCHORED EVENT facts — things the user
did at a specific time (a trip, a race, a surgery, a move, a job change, a
purchase, a wedding, a concert). Put these in category "event" (health/career
events may keep their own category). For these facts set "event_time"
("YYYY-MM") AND embed the date or month inside the value itself (e.g. "a
two-week trip to Japan in May 2024", "arthroscopic knee surgery in February
2023"), so that "When did I ...?" questions can be answered from the value.

If the persona's region is outside the US, use locally appropriate names,
street addresses, schools, employers, and brands (e.g. for Beijing use Chinese
street names and local companies).

Do NOT include derived facts or a "source_facts" field: derived facts are
computed by the pipeline, not by you.
Return ONLY a JSON object with one key: {{"facts": [...]}}."""


# --------------------------------------------------------------------------
# LLM-based derivation
# --------------------------------------------------------------------------

_DERIVE_PROMPT = """You are building a synthetic user profile for a memory \
benchmark. A set of atomic facts about the user already exists. Derive NEW \
facts that a person (or a memory system) can logically conclude by COMBINING \
TWO OR MORE of the existing facts.

Existing atomic facts (id | category | predicate: value):
{fact_list}

Propose 2-4 new derived facts. Each one MUST:
- follow from at least 2 of the listed facts, citing them in "source_fact_ids";
- be genuinely inferable: the conclusion must be forced (or near-forced) by \
the sources, not speculation. Examples: "likes hot drinks" + "likes milk" \
imply "likes hot milk"; "home in Springfield" + "office in Lakewood" imply \
a commute "from Springfield to Lakewood"; "ran the Chicago marathon" + "has \
a torn knee ligament" is NOT a valid inference;
- be a NEW fact — never a restatement of any single source fact;
- have a "value" that is a short noun phrase (at most 8 words) naming the \
concluded fact, never a clause with a finite verb: "from Springfield to \
Lakewood" and "retirement in 2026 after 16 contribution years" are values; \
"commutes from Springfield to Lakewood" and "plans to retire after 16 years \
of max contributions" are clauses and will be rejected — a clause-shaped \
value cannot be the clean gold answer of a recall question;
- have a first-person "surface_form" and a "category" from: \
address/health/preference/career/education/relationship/finance/routine/\
event/hobby/social/other.

Return ONLY a JSON object with one key: {{"derived": [...]}}; each element:
{{"predicate": "<short relation phrase>", "value": "<the new fact value>", \
"category": "<one of the categories>", "surface_form": "<first-person \
sentence>", "source_fact_ids": ["<id>", "<id>", ...], \
"sensitivity": "low|medium|high", "event_time": "YYYY-MM" or null}}"""

_DERIVE_VERIFY_PROMPT = """A memory system knows these facts about a user:
{source_list}

It claims it can now conclude: {conclusion}

Does the conclusion logically follow from the listed facts — strictly, \
without external knowledge, speculation, or probability hedging? And is the \
concluded value phrased as a concise noun phrase (like "from Springfield to \
Lakewood"), not a clause with a finite verb (like "plans to retire after 16 \
years")? Answer ONLY JSON: {{"valid": true/false, "reason": "..."}}"""

_ISO_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _validate_profile_facts(facts: list[AtomicFact]) -> None:
    """Enforce the hard requirements the facts prompt states.

    The prompt demands full category coverage and a rich set of past
    time-anchored events, but the model does not always comply; F3 (category
    forgetting) and F5 (event-time forgetting) silently degrade when it does
    not. A violating profile raises and is retried/skipped, consistent with
    the rest of the pipeline. A malformed event_time is normalized to None
    (it only loses the fact's time anchor); an event_time at or after the
    dialogue timeline start is a hard failure.
    """
    if len(facts) < 30:
        raise ValueError(f"only {len(facts)} atomic facts (need at least 30)")
    for f in facts:
        if not (f.predicate.strip() and f.value.strip() and f.surface_form.strip()):
            raise ValueError(f"fact {f.fact_id} has an empty predicate/value/surface")
        if f.event_time is not None:
            if not _ISO_MONTH_RE.match(f.event_time):
                f.event_time = None
            elif f.event_time >= _REFERENCE_MONTH:
                raise ValueError(
                    f"fact {f.fact_id} event_time {f.event_time} is not before "
                    f"the dialogue timeline start {_REFERENCE_MONTH}"
                )
    missing = set(FactCategory) - {f.category for f in facts}
    if missing:
        raise ValueError(
            f"categories not covered: {sorted(c.value for c in missing)}"
        )
    anchored = sum(1 for f in facts if f.event_time)
    if anchored < 8:
        raise ValueError(
            f"only {anchored} past time-anchored facts (need at least 8)"
        )


def _validate_derived_candidate(
    item: Any,
    fmap: dict[str, AtomicFact],
    existing: set[tuple[str, str]],
) -> AtomicFact | None:
    """Structurally validate one LLM-proposed derived fact; None when invalid.

    Rejects: missing fields, fewer than 2 sources, unknown source ids, source
    facts that are themselves derived, invalid categories, verbatim
    restatements of a single source, and duplicates of existing facts.
    """
    try:
        category = FactCategory(item["category"])
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(item, dict):
        return None
    sources = item.get("source_fact_ids")
    if not isinstance(sources, list) or len(sources) < 2:
        return None
    if any(s not in fmap for s in sources) or any(fmap[s].is_derived for s in sources):
        return None
    predicate = str(item.get("predicate", "")).strip()
    value = str(item.get("value", "")).strip()
    surface = str(item.get("surface_form", "")).strip()
    if not (predicate and value and surface):
        return None
    # The value is a probe gold answer: it must be a concise noun phrase, not
    # a clause with a finite verb (those make every recall question either
    # awkward or ungradeable).
    if not is_clean_value(value) or len(value.split()) > 8:
        return None
    if (predicate, value) in existing:
        return None
    # A derived fact must not restate any single source (verbatim value).
    if any(value.lower() == fmap[s].value.lower() for s in sources):
        return None
    try:
        sensitivity = Sensitivity(item.get("sensitivity", "medium"))
    except ValueError:
        sensitivity = Sensitivity.MEDIUM
    event_time = item.get("event_time")
    if event_time is not None and not (isinstance(event_time, str)
                                       and _ISO_MONTH_RE.match(event_time)):
        event_time = None
    return AtomicFact(
        fact_id="",  # assigned by the caller
        category=category,
        subject=fmap[sources[0]].subject,
        predicate=predicate,
        value=value,
        surface_form=surface,
        event_time=event_time,
        sensitivity=sensitivity,
        source_facts=[str(s) for s in sources],
    )


def _verify_derived_candidate(
    candidate: AtomicFact, fmap: dict[str, AtomicFact], llm: BaseLLM,
    temperature: float,
) -> bool:
    """Ask the LLM whether the candidate strictly follows from its sources."""
    source_lines = "\n".join(
        f"- {fmap[s].surface_form}" for s in candidate.source_facts
    )
    raw = llm.chat_json(
        [{"role": "user", "content": _DERIVE_VERIFY_PROMPT.format(
            source_list=source_lines,
            conclusion=f"{candidate.predicate}: {candidate.value} — "
                       f"({candidate.surface_form})")}],
        temperature=temperature,
        task="derive_verify",
    )
    return bool(raw.get("valid", False))


def _llm_derive_facts(
    facts: list[AtomicFact],
    llm: BaseLLM,
    config: dict[str, Any],
    profile_id: str,
    name: str,
) -> list[AtomicFact]:
    """LLM-based derivation.

    Asks the model to combine 2+ existing atomic facts into new facts, then
    verifies each candidate's inferability with a second LLM call. Candidates
    that fail structural validation or the verifier are dropped; the whole
    call is retried (bounded) while nothing valid survives, and if the LLM
    still produces nothing inferable the profile is failed (the caller's
    retry/skip policy handles it) instead of emitting a canned derivation.

    Every returned fact has `source_facts` set and its sources' `derivations`
    back-filled.
    """
    fmap = {f.fact_id: f for f in facts if not f.is_derived}
    existing = {(f.predicate, f.value) for f in facts}
    gen_cfg = config.get("generation", {})
    temperature = gen_cfg.get("temperature", 0.9)
    attempts = gen_cfg.get("max_retries", 3)
    fact_list = "\n".join(
        f"- {f.fact_id} | {f.category.value} | {f.predicate}: {f.value}"
        for f in facts if not f.is_derived
    )
    last_err: Exception | None = None

    for _ in range(attempts):
        try:
            raw = llm.chat_json(
                [{"role": "user", "content": _DERIVE_PROMPT.format(fact_list=fact_list)}],
                temperature=temperature,
                task="derive_facts",
            )
        except Exception as e:  # noqa: BLE001 - retry covers all call failures
            last_err = e
            continue
        candidates: list[AtomicFact] = []
        seen: set[tuple[str, str]] = set(existing)
        for item in _json_list(raw, "derived"):
            cand = _validate_derived_candidate(item, fmap, seen)
            if cand is None:
                continue
            seen.add((cand.predicate, cand.value))
            candidates.append(cand)
        verified: list[AtomicFact] = []
        for cand in candidates:
            try:
                if _verify_derived_candidate(cand, fmap, llm, temperature):
                    verified.append(cand)
            except Exception as e:  # noqa: BLE001 - retried with the next attempt
                last_err = e
        if verified:
            derived: list[AtomicFact] = []
            next_id = len(facts)
            for cand in verified:
                fact = cand.model_copy(update={
                    "fact_id": f"{profile_id}:f{next_id:02d}",
                    "subject": name,
                })
                next_id += 1
                derived.append(fact)
                for sid in fact.source_facts:
                    fmap[sid].derivations.append(fact.fact_id)
            return derived
        last_err = ValueError("no candidate passed structural validation/verification")
    raise RuntimeError(
        f"LLM derivation produced no valid derived facts for {profile_id} "
        f"(last error: {last_err})"
    )


def _json_list(raw: Any, key: str) -> list[Any]:
    """Unwrap an array-prompt response.

    json_object mode (enforced for proxy models) cannot return a bare array,
    so the model wraps it in a single-key object ({"facts": [...]}); the
    direct DeepSeek path still returns bare arrays. Accept both shapes.
    """
    if isinstance(raw, dict):
        raw = raw.get(key, [])
    return raw if isinstance(raw, list) else []


def _llm_profile(
    index: int, seed_attrs: dict[str, str], llm: BaseLLM, config: dict[str, Any]
) -> UserProfile:
    profile_id = f"profile-{index:04d}"
    name = seed_attrs["name"]
    n_facts = 60  # within the 30-80 range
    gen_cfg = config.get("generation", {})
    raw = llm.chat_json(
        [
            {
                "role": "user",
                "content": _FACTS_PROMPT.format(
                    seed_attrs=seed_attrs, n_facts=n_facts, name=name,
                    present_month=_BASE_DATE.strftime("%B %Y"),
                    ref_month=_REFERENCE_MONTH,
                ),
            }
        ],
        temperature=gen_cfg.get("temperature", 0.9),
        task="profile_facts",
    )
    facts: list[AtomicFact] = []
    for j, item in enumerate(_json_list(raw, "facts")):
        # A model-invented category label (e.g. "derived") is a hard failure:
        # silently re-tagging the fact as "other" would mislabel its
        # category-driven probes (F3, preservation), so it raises and the
        # profile is retried/skipped instead of degraded.
        category = FactCategory(item["category"])
        facts.append(
            AtomicFact(
                fact_id=f"{profile_id}:f{j:02d}",
                category=category,
                subject=item.get("subject", name),
                predicate=item["predicate"],
                value=str(item["value"]),
                surface_form=item["surface_form"],
                event_time=item.get("event_time"),
                sensitivity=item.get("sensitivity", "low"),
            )
        )
    # LLM-based derivation: the model combines 2+ atomic facts into new
    # inferable facts (with `source_facts` and back-filled `derivations`).
    _validate_profile_facts(facts)
    facts.extend(_llm_derive_facts(facts, llm, config, profile_id, name))
    summary = (
        f"{name}, a {seed_attrs['age']}-year-old {seed_attrs['gender']} "
        f"{seed_attrs['occupation']} from {seed_attrs['region']}, "
        f"{seed_attrs['family_structure']}."
    )
    return UserProfile(
        profile_id=profile_id,
        persona_summary=summary,
        demographics={"name": name, **seed_attrs},
        facts=facts,
    )
