from pathlib import Path
import re
from rapidfuzz import fuzz

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OBO_PATH = BASE_DIR / "data" / "hp.obo"


def load_hpo_data(obo_path=DEFAULT_OBO_PATH):
    all_terms = []
    current_term = {"synonyms": []}

    with open(obo_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                if "id" in current_term and "name" in current_term:
                    all_terms.append(current_term)

                current_term = {"synonyms": []}

            elif line.startswith("id:"):
                current_term["id"] = line.split("id: ")[1]

            elif line.startswith("name:"):
                current_term["name"] = line.split("name: ")[1]

            elif line.startswith("synonym:"):
                current_term["synonyms"].append(
                    line.split('"')[1]
                )

    if "id" in current_term and "name" in current_term:
        all_terms.append(current_term)

    return all_terms


def search_symptoms(user_input, terms):
    matching_terms = []

    clean_input = re.sub(r"[^a-z0-9 ]+", " ", user_input.lower()).strip()
    candidate_words = clean_input.split()

    if not candidate_words:
        return matching_terms

    for term in terms:
        name = re.sub(r"[^a-z0-9 ]+", " ", term["name"].lower()).strip()

        # Check the official HPO name and all synonyms
        phrases = [name] + [
            re.sub(r"[^a-z0-9 ]+", " ", synonym.lower()).strip()
            for synonym in term["synonyms"]
        ]

        best_score = 0

        for phrase in phrases:
            phrase_words = phrase.split()

            matching_words = len(set(candidate_words) & set(phrase_words))

            # If NONE of the words match, reject it.
            if matching_words == 0:
                continue

            candidate_coverage = matching_words / len(set(candidate_words))
            phrase_coverage = matching_words / len(set(phrase_words))

            # Multi-word paraphrases may share one defining token, but a
            # single-word query must be an exact token match.
            minimum_candidate_coverage = 1.0 if len(candidate_words) == 1 else 0.5
            if len(candidate_words) == 1 and len(phrase_words) > 1:
                continue
            if candidate_coverage < minimum_candidate_coverage or phrase_coverage < 0.5:
                continue

            # Now use RapidFuzz only after the word-overlap
            # requirement has been satisfied.
            similarity = fuzz.token_sort_ratio(
                clean_input,
                phrase
            )

            # Single words need an exact token match; fuzzy matching a generic
            # word creates many clinically unrelated HPO results.
            if len(candidate_words) == 1 and candidate_words[0] not in phrase_words:
                continue

            if similarity < (60 if len(candidate_words) == 1 else 80):
                continue

            # Prefer exact phrases and phrases that cover more of the HPO name.
            weighted_score = similarity * (0.7 + 0.3 * phrase_coverage)

            if weighted_score > best_score:
                best_score = weighted_score

        if best_score > 0:
            matching_terms.append(
                (term, best_score)
            )

    # Best matches first
    matching_terms.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return matching_terms