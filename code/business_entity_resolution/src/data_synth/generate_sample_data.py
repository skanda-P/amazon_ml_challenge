"""
Generates a small, self-contained synthetic dataset that mimics the shape
and noise patterns described in the challenge problem statement (name
abbreviations/typos/reordering, address abbreviations/missing components/
landmarks, multi-country records, a test-only unseen country).

This is NOT the real challenge dataset (which is provided by the
organizers) -- it exists purely so the pipeline in this submission can be
demonstrated end-to-end and self-validated before being pointed at the
real `dataset/train` and `dataset/test` directories.

No external data or services are used -- everything is generated locally
from small in-code word lists.
"""
from __future__ import annotations

import csv
import os
import random

FIRST_WORDS = [
    "Bluebird", "Sunrise", "Zenith", "Global", "Pioneer", "Silverline", "Crimson",
    "Northstar", "Evergreen", "Falcon", "Golden", "Riverside", "Metro", "Prime",
    "Horizon", "Summit", "Coastal", "Imperial", "Vanguard", "Meridian", "Kumar",
    "Sharma", "Patel", "Rao", "Ashoka", "Ganges", "Himalaya", "Lotus", "Saffron",
    "Bharat", "Nova", "Apex", "Sterling", "Atlas", "Orion", "Beacon", "Cascade",
]
BUSINESS_TYPES = [
    "Logistics", "Foods", "Traders", "Micro Systems", "Textiles", "Motors",
    "Pharmaceuticals", "Electronics", "Consulting", "Bakery", "Hardware",
    "Constructions", "Agro", "Retail", "Solutions", "Enterprises", "Exports",
    "Technologies", "Furnishings", "Beverages",
]
LEGAL_SUFFIXES_US = ["Inc", "LLC", "Corp", "Co"]
LEGAL_SUFFIXES_IN = ["Pvt Ltd", "Ltd", "and Sons", "and Co"]
LEGAL_SUFFIXES_FR = ["SARL", "SA", "SAS"]

STREETS_US = ["Maple", "Oak", "Main", "Sunset", "Lincoln", "Park", "Highland", "Cedar"]
STREETS_IN = ["MG", "Nehru", "Gandhi", "Station", "Church", "College", "Market", "Ring"]
STREETS_FR = ["Victor Hugo", "de la Paix", "des Lilas", "du Commerce", "Voltaire"]

US_CITIES = [("Springfield", "IL"), ("Austin", "TX"), ("Fresno", "CA"), ("Dayton", "OH"), ("Reno", "NV")]
IN_CITIES = [("Pune", "MH"), ("Indore", "MP"), ("Coimbatore", "TN"), ("Jaipur", "RJ"), ("Nagpur", "MH")]
FR_CITIES = ["Lyon", "Nantes", "Toulouse", "Lille", "Rennes"]

LANDMARKS = ["Near SBI ATM", "Opp City Mall", "Behind Central Park", "Next to Post Office"]

LEGAL_ABBR_SWAP = {"Corporation": "Corp", "Limited": "Ltd", "Private": "Pvt", "Incorporated": "Inc"}


def _rand_name(rng: random.Random, country: str) -> str:
    core = f"{rng.choice(FIRST_WORDS)} {rng.choice(BUSINESS_TYPES)}"
    if country == "US":
        suffix = rng.choice(LEGAL_SUFFIXES_US)
    elif country == "India":
        suffix = rng.choice(LEGAL_SUFFIXES_IN)
    else:
        suffix = rng.choice(LEGAL_SUFFIXES_FR)
    return f"{core} {suffix}"


def _rand_address(rng: random.Random, country: str) -> str:
    house = rng.randint(1, 999)
    if country == "US":
        city, state = rng.choice(US_CITIES)
        street = rng.choice(STREETS_US)
        zipc = rng.randint(10000, 99999)
        return f"{house} {street} St, {city}, {state} {zipc}"
    elif country == "India":
        city = rng.choice(IN_CITIES)[0]
        street = rng.choice(STREETS_IN)
        pin = rng.randint(100000, 999999)
        return f"{house}, {street} Road, {city} - {pin}"
    else:
        city = rng.choice(FR_CITIES)
        street = rng.choice(STREETS_FR)
        postal = rng.randint(10000, 99999)
        return f"{house} Rue {street}, {postal} {city}"


def _typo(rng: random.Random, s: str) -> str:
    if len(s) < 4:
        return s
    i = rng.randint(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _noisy_name(rng: random.Random, name: str) -> str:
    tokens = name.split()
    r = rng.random()
    if r < 0.25:
        for full, abbr in LEGAL_ABBR_SWAP.items():
            name = name.replace(full, abbr)
        tokens = name.split()
    if r < 0.5 and len(tokens) > 2:
        # word-order transposition of the first two tokens
        tokens[0], tokens[1] = tokens[1], tokens[0]
    name2 = " ".join(tokens)
    if rng.random() < 0.4:
        name2 = _typo(rng, name2)
    if rng.random() < 0.15:
        name2 = name2.replace("and", "&")
    return name2


ADDR_ABBR_SWAP = {"Street": "St", "Road": "Rd", "Avenue": "Ave"}


def _noisy_address(rng: random.Random, addr: str) -> str:
    a = addr
    r = rng.random()
    for full, abbr in ADDR_ABBR_SWAP.items():
        if full in a and rng.random() < 0.5:
            a = a.replace(full, abbr)
    if r < 0.3:
        # drop the postal/pin code component
        a = a.split(",")[0] if "," in a else a
    if r < 0.2:
        a = f"{a}, {rng.choice(LANDMARKS)}"
    if rng.random() < 0.3:
        a = _typo(rng, a)
    return a


def generate(out_dir: str, n_train: int = 400, n_test: int = 200, seed: int = 13):
    rng = random.Random(seed)

    def make_split(n_base: int, split_countries: list, include_singletons=True):
        s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
        s1_counter = s2_counter = s3_counter = 1
        for _ in range(n_base):
            country = rng.choice(split_countries)
            name = _rand_name(rng, country)
            addr = _rand_address(rng, country)
            s1_id = f"S1-{s1_counter:05d}"
            s1_counter += 1
            s1_rows.append([s1_id, name, addr, country])

            matched_ids = []
            # 0-3 noisy matches in source2
            n_s2 = rng.choices([0, 1, 2], weights=[0.25, 0.6, 0.15])[0]
            for _ in range(n_s2):
                s2_id = f"S2-{s2_counter:05d}"
                s2_counter += 1
                s2_rows.append([s2_id, _noisy_name(rng, name), _noisy_address(rng, addr), country])
                matched_ids.append(s2_id)
            # 0-2 noisy matches in source3
            n_s3 = rng.choices([0, 1, 2], weights=[0.4, 0.5, 0.1])[0]
            for _ in range(n_s3):
                s3_id = f"S3-{s3_counter:05d}"
                s3_counter += 1
                s3_rows.append([s3_id, _noisy_name(rng, name), _noisy_address(rng, addr), country])
                matched_ids.append(s3_id)

            gt_rows.append([s1_id, ",".join(matched_ids)])

        # add distractor-only records (no match to any S1 entity) to both
        # vendor sources, including near-duplicate-looking-but-different
        # businesses to stress precision.
        n_distractors_s2 = max(1, n_base // 5)
        n_distractors_s3 = max(1, n_base // 6)
        for _ in range(n_distractors_s2):
            country = rng.choice(split_countries)
            s2_rows.append([f"S2-{s2_counter:05d}", _rand_name(rng, country), _rand_address(rng, country), country])
            s2_counter += 1
        for _ in range(n_distractors_s3):
            country = rng.choice(split_countries)
            s3_rows.append([f"S3-{s3_counter:05d}", _rand_name(rng, country), _rand_address(rng, country), country])
            s3_counter += 1

        # "confusable" distractors: reuse an EXISTING S1 business name almost
        # verbatim but pair it with a completely different address (a
        # different real-world business that happens to share a name/brand
        # pattern -- e.g. two unrelated "Sunrise Foods Pvt Ltd"). These are
        # the genuinely hard negatives that separate a naive name-similarity
        # threshold from an address-aware, precision-oriented decoder.
        n_confusable = max(1, n_base // 4)
        for _ in range(n_confusable):
            base = rng.choice(s1_rows)
            _, base_name, _, base_country = base
            other_country = rng.choice(split_countries)
            fake_addr = _rand_address(rng, other_country)
            target_list = s2_rows if rng.random() < 0.5 else s3_rows
            if target_list is s2_rows:
                cid = f"S2-{s2_counter:05d}"
                s2_counter += 1
            else:
                cid = f"S3-{s3_counter:05d}"
                s3_counter += 1
            target_list.append([cid, _noisy_name(rng, base_name), fake_addr, other_country])

        rng.shuffle(s1_rows)
        rng.shuffle(s2_rows)
        rng.shuffle(s3_rows)
        return s1_rows, s2_rows, s3_rows, gt_rows

    def write_tsv(path, header, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    # --- train: US + India only, as stated in the problem ---
    tr_s1, tr_s2, tr_s3, tr_gt = make_split(n_train, ["US", "India"])
    write_tsv(os.path.join(out_dir, "train", "train_source1.tsv"),
              ["entity_id", "business_name", "business_address", "country"], tr_s1)
    write_tsv(os.path.join(out_dir, "train", "train_source2.tsv"),
              ["entity_id", "business_name", "business_address", "country"], tr_s2)
    write_tsv(os.path.join(out_dir, "train", "train_source3.tsv"),
              ["entity_id", "business_name", "business_address", "country"], tr_s3)
    write_tsv(os.path.join(out_dir, "train", "train_ground_truth.tsv"),
              ["source1_entity_id", "matched_entity_ids"], tr_gt)

    # --- test: US + India + an UNSEEN country (France) ---
    te_s1, te_s2, te_s3, te_gt = make_split(n_test, ["US", "India", "France"])
    write_tsv(os.path.join(out_dir, "test", "test_source1.tsv"),
              ["entity_id", "business_name", "business_address", "country"], te_s1)
    write_tsv(os.path.join(out_dir, "test", "test_source2.tsv"),
              ["entity_id", "business_name", "business_address", "country"], te_s2)
    write_tsv(os.path.join(out_dir, "test", "test_source3.tsv"),
              ["entity_id", "business_name", "business_address", "country"], te_s3)
    # test ground truth is withheld in the real challenge; we still save a
    # hidden copy for OUR OWN local scoring/demo purposes only (never used
    # by the pipeline itself, exactly mirroring "hold out a validation
    # split" guidance in the problem statement).
    write_tsv(os.path.join(out_dir, "test", "_hidden_test_ground_truth_FOR_DEMO_ONLY.tsv"),
              ["source1_entity_id", "matched_entity_ids"], te_gt)

    print(f"Sample data written to {out_dir}")
    print(f"  train: S1={len(tr_s1)} S2={len(tr_s2)} S3={len(tr_s3)}")
    print(f"  test:  S1={len(te_s1)} S2={len(te_s2)} S3={len(te_s3)} (+ France, unseen in train)")
