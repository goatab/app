# application/extractor.py
from infrastructure import search_symptoms
import re


def extract_symptoms(user_input, hpo_terms):
    clauses = re.split(r"[,;.!?]|\band\b", user_input.lower())

    # Words that usually don't carry useful symptom information
    stop_words = {
        "a", "an", "the",
        "and", "or",
        "patient", "presents",
        "with", "has", "have", "had",
        "is", "are", "was", "were",
        "years",
        "in", "on", "of", "to", "for",
        "from", "at", "by",
        "as", "into", "without",
        "symptom", "symptoms",
        "male", "female", "boy", "girl", "man", "woman",
        "year", "years", "old", "severe", "mild", "moderate",
    }

    candidates = []

    # -------------------------
    # Generate candidates
    # -------------------------

    # Keep only contiguous spans. Disconnected pairs turn unrelated words
    # into plausible-looking symptoms.
    negation_active = False
    for clause in clauses:
        if re.search(r"\b(?:presents?|reports?|has|have|shows?)\b", clause):
            negation_active = False

        if re.search(r"\b(?:no|without|denies|denied|negative for)\b", clause):
            negation_active = True

        if negation_active:
            continue

        clean_words = [
            word for word in re.findall(r"[a-z]+", clause)
            if word not in stop_words
        ]
        for span_length in range(1, min(4, len(clean_words) + 1)):
            for start in range(len(clean_words) - span_length + 1):
                candidates.append(" ".join(clean_words[start:start + span_length]))

    # ------------------------------------------------
    # Find the BEST candidate for each HPO term
    # ------------------------------------------------

    best_matches = {}

    for candidate in candidates:

        results = search_symptoms(candidate, hpo_terms)

        if not results:
            continue

        # search_symptoms already ranks the HPO matches for this candidate.
        term, similarity = results[0]

        candidate_length = len(candidate.split())

        # Longer exact spans are more informative than isolated words.
        length_bonus = (candidate_length - 1) * 3

        final_score = similarity + length_bonus

        term_id = term["id"]

        # Only keep the best candidate for this HPO term
        if (
            term_id not in best_matches
            or final_score > best_matches[term_id]["score"]
        ):
            best_matches[term_id] = {
                "term": term,
                "score": final_score,
                "candidate": candidate,
            }

    # ---------------------------------------------
    # Convert dictionary into a list
    # ---------------------------------------------

    matched_terms = list(best_matches.values())

    # Do not return a generic sub-span when a longer span already produced a
    # more specific HPO concept (for example, pain within joint pain).
    matched_terms = [
        match for match in matched_terms
        if not any(
            len(match["candidate"].split()) < len(other["candidate"].split())
            and set(match["candidate"].split()).issubset(other["candidate"].split())
            and other["score"] >= match["score"]
            for other in matched_terms
        )
    ]

    # Highest score first
    matched_terms.sort(
        key=lambda x: (
            x["score"],
            len(x["candidate"].split())
        ),
        reverse=True
    )

    # ---------------------------------------------
    # Return format expected by app.py
    # ---------------------------------------------

    return [
        (match["term"], match["score"])
        for match in matched_terms
    ]