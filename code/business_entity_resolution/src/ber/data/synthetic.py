"""Synthetic benchmark data generator simulating official competition data characteristics."""
import csv
import os
import random
from typing import Dict, List, Tuple

BASE_BUSINESSES = [
    # US businesses
    ("Walmart Supercenter", "702 SW 8th St, Bentonville, AR 72716", "US"),
    ("Starbucks Coffee", "2401 Utah Ave S, Seattle, WA 98134", "US"),
    ("Target Store", "1000 Nicollet Mall, Minneapolis, MN 55403", "US"),
    ("Home Depot", "2455 Paces Ferry Rd NW, Atlanta, GA 30339", "US"),
    ("Costco Wholesale", "999 Lake Dr, Issaquah, WA 98027", "US"),
    ("CVS Pharmacy", "One CVS Dr, Woonsocket, RI 02895", "US"),
    ("Walgreens Pharmacy", "108 Wilmot Rd, Deerfield, IL 60015", "US"),
    ("Apex Tech Solutions", "1600 Amphitheatre Pkwy, Mountain View, CA 94043", "US"),
    ("Kroger Supermarket", "1014 Vine St, Cincinnati, OH 45202", "US"),
    ("Chevron Gas Station", "6001 Bollinger Canyon Rd, San Ramon, CA 94583", "US"),
    # India businesses
    ("Tata Consultancy Services", "Nirmal Building, 9th Floor, Nariman Point, Mumbai, Maharashtra 400021", "India"),
    ("Infosys Technologies Ltd", "Plot No. 44, Electronics City, Hosur Road, Bangalore, Karnataka 560100", "India"),
    ("Reliance Retail Mart", "Near City Center Mall, MG Road, Pune, Maharashtra 411001", "India"),
    ("State Bank of India", "State Bank Bhavan, Madame Cama Road, Nariman Point, Mumbai 400021", "India"),
    ("HDFC Bank Branch", "HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai 400013", "India"),
    ("Wipro Limited", "Doddakannelli, Sarjapur Road, Bangalore, Karnataka 560035", "India"),
    ("Sharma Sweet House", "Shop No. 12, Main Bazaar, Chandni Chowk, Delhi 110006", "India"),
    ("Apollo Pharmacy", "Opposite Metro Pillar 45, Indiranagar 100ft Road, Bangalore 560038", "India"),
    ("Larsen and Toubro Infotech", "L&T House, Ballard Estate, Mumbai 400001", "India"),
    ("Aggarwal Sweets and Snacks", "Plot 5, Sector 15 Market, Faridabad, Haryana 121007", "India"),
    # France businesses (for test split)
    ("Carrefour Hypermarche", "Boulevard Saint-Germain, Paris 75005", "France"),
    ("BNP Paribas Banque", "16 Boulevard des Italiens, Paris 75009", "France"),
    ("TotalEnergies Station", "2 Place Jean Millier, La Defense, Courbevoie 92400", "France"),
    ("Boulangerie Paul", "Avenue de l'Opera, Paris 75001", "France"),
]


def corrupt_business_name(name: str, rng: random.Random) -> str:
    """Applies realistic name noise: abbreviations, legal suffixes, typos, DBA."""
    n = name
    p = rng.random()

    # Suffix modifications
    if "Ltd" in n or "Limited" in n:
        if rng.random() < 0.5:
            n = n.replace("Technologies Ltd", "Tech Pvt Ltd").replace("Limited", "Ltd")
        else:
            n = n.replace("Ltd", "Private Limited")

    if "Corporation" in n or "Corp" in n:
        n = n.replace("Corporation", "Corp.")

    # Punctuation & vs and
    if " and " in n:
        if rng.random() < 0.5:
            n = n.replace(" and ", " & ")
    elif " & " in n:
        n = n.replace(" & ", " and ")

    # DBA injection
    if rng.random() < 0.25 and "(" not in n:
        tokens = n.split()
        n = f"{n} ({tokens[0]} Express)"

    # Minor typo injection
    if rng.random() < 0.25 and len(n) > 5:
        idx = rng.randint(2, len(n) - 3)
        c = n[idx]
        if c.isalpha():
            n = n[:idx] + c + n[idx:]  # Double letter

    return n


def corrupt_address(addr: str, rng: random.Random) -> str:
    """Applies realistic address noise: street abbreviations, landmarks, component drops."""
    a = addr

    # Street abbreviations
    replacements = [
        (" Road", " Rd"),
        (" Street", " St"),
        (" Avenue", " Ave"),
        (" Boulevard", " Blvd"),
        (" Marg", " Road"),
    ]
    for orig, repl in replacements:
        if orig in a and rng.random() < 0.5:
            a = a.replace(orig, repl)

    # Landmark insertion
    if "Near" not in a and "Opposite" not in a and rng.random() < 0.3:
        a = f"Near City Hospital, {a}"

    # Drop state / postal code occasionally
    if rng.random() < 0.2:
        parts = a.split(",")
        if len(parts) > 2:
            a = ",".join(parts[:-1])

    return a


def generate_synthetic_benchmark(
    base_dir: str = "dataset",
    train_size: int = 100,
    test_size: int = 50,
    seed: int = 42,
) -> None:
    """Generates synthetic training and test TSVs adhering to the exact dataset schema."""
    rng = random.Random(seed)

    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)

    # Filter base pools: Train uses US and India; Test includes France
    train_base = [b for b in BASE_BUSINESSES if b[2] in ("US", "India")]
    test_base = BASE_BUSINESSES

    # --- Generate Train Split ---
    s1_train, s2_train, s3_train = [], [], []
    ground_truth = []

    s2_id_counter = 1
    s3_id_counter = 1

    for i in range(1, train_size + 1):
        s1_id = f"S1-{i:05d}"
        template_name, template_addr, template_country = rng.choice(train_base)

        s1_train.append([s1_id, template_name, template_addr, template_country])

        # 35% chance of being a singleton (empty gold)
        if rng.random() < 0.35:
            ground_truth.append([s1_id, ""])
            continue

        # Non-singleton: 1 to 3 matches
        num_matches = 1 if rng.random() < 0.65 else (2 if rng.random() < 0.85 else 3)
        matched_ids = []

        for _ in range(num_matches):
            dest_source = 2 if rng.random() < 0.6 else 3
            corrupted_n = corrupt_business_name(template_name, rng)
            corrupted_a = corrupt_address(template_addr, rng)

            if dest_source == 2:
                pid = f"S2-{s2_id_counter:05d}"
                s2_id_counter += 1
                s2_train.append([pid, corrupted_n, corrupted_a, template_country])
            else:
                pid = f"S3-{s3_id_counter:05d}"
                s3_id_counter += 1
                s3_train.append([pid, corrupted_n, corrupted_a, template_country])

            matched_ids.append(pid)

        ground_truth.append([s1_id, ",".join(matched_ids)])

    # Add distractors (S2 and S3 records that match NO S1)
    for _ in range(int(train_size * 0.25)):
        template_name, template_addr, template_country = rng.choice(train_base)
        pid = f"S2-{s2_id_counter:05d}"
        s2_id_counter += 1
        s2_train.append([pid, f"{template_name} Branch", f"99 Other Rd, {template_country}", template_country])

    # Write Train TSVs
    def write_tsv(path, rows, headers):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar=None, lineterminator="\n")
            w.writerow(headers)
            for r in rows:
                w.writerow(r)

    headers_record = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(os.path.join(base_dir, "train", "train_source1.tsv"), s1_train, headers_record)
    write_tsv(os.path.join(base_dir, "train", "train_source2.tsv"), s2_train, headers_record)
    write_tsv(os.path.join(base_dir, "train", "train_source3.tsv"), s3_train, headers_record)
    write_tsv(
        os.path.join(base_dir, "train", "train_ground_truth.tsv"),
        ground_truth,
        ["source1_entity_id", "matched_entity_ids"],
    )

    # --- Generate Test Split ---
    s1_test, s2_test, s3_test = [], [], []
    for i in range(1, test_size + 1):
        s1_id = f"S1-{i:05d}"
        template_name, template_addr, template_country = rng.choice(test_base)
        s1_test.append([s1_id, template_name, template_addr, template_country])

        # Generate corresponding dirty records in pool
        if rng.random() > 0.30:  # Not singleton
            dest_source = 2 if rng.random() < 0.5 else 3
            corrupted_n = corrupt_business_name(template_name, rng)
            corrupted_a = corrupt_address(template_addr, rng)
            if dest_source == 2:
                pid = f"S2-{i:05d}"
                s2_test.append([pid, corrupted_n, corrupted_a, template_country])
            else:
                pid = f"S3-{i:05d}"
                s3_test.append([pid, corrupted_n, corrupted_a, template_country])

    # Add distractors to test pool
    for k in range(test_size + 1, test_size + 20):
        template_name, template_addr, template_country = rng.choice(test_base)
        s2_test.append([f"S2-{k:05d}", f"{template_name} Alt", f"120 Park Ave, {template_country}", template_country])

    write_tsv(os.path.join(base_dir, "test", "test_source1.tsv"), s1_test, headers_record)
    write_tsv(os.path.join(base_dir, "test", "test_source2.tsv"), s2_test, headers_record)
    write_tsv(os.path.join(base_dir, "test", "test_source3.tsv"), s3_test, headers_record)
