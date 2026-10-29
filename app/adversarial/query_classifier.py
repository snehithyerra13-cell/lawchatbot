"""Adversarial Query Classifier & Premise Verifier for ConstitutionGPT.

Detects:
1. False Premises & Fabricated Assumptions (e.g. "Under Article 14, how are hereditary titles granted?")
2. Article Misattributions (e.g. "Why does Article 21 grant the President pardon powers?")
3. Ambiguous and Underspecified Inquiries (e.g. "Who has power to issue ordinances?")
4. Multi-hop and Multi-Article Synthesis Requirements (e.g. "Compare Article 32 and Article 226 writ powers")
5. Temporal, Dynamic & Post-Corpus Inquiries (e.g. "What amendments passed in 2026?")
6. Truly Out-of-Domain Non-Constitutional Inquiries (e.g. "Capital of France", "Python code")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Query Types
QUERY_TYPE_EXACT_ARTICLE = "exact_article"
QUERY_TYPE_CLAUSE_LOOKUP = "clause_lookup"
QUERY_TYPE_MULTI_HOP = "multi_hop"
QUERY_TYPE_FALSE_PREMISE = "false_premise"
QUERY_TYPE_AMBIGUOUS = "ambiguous"
QUERY_TYPE_TEMPORAL_DYNAMIC = "temporal_dynamic"
QUERY_TYPE_OUT_OF_DOMAIN = "out_of_domain"
QUERY_TYPE_GENERAL_SEMANTIC = "general_semantic"

# Canonical Landmark Mapping: Article -> Expected Core Subject
LANDMARK_SUBJECTS: Dict[str, Dict[str, Any]] = {
    "PREAMBLE": {"keywords": ["we, the people", "sovereign", "socialist", "secular", "democratic", "republic", "justice", "liberty", "equality", "fraternity"]},
    "1": {"subject": "Name and territory of the Union", "keywords": ["bharat", "union of states", "territory"]},
    "12": {"subject": "Definition of State", "keywords": ["state", "government", "parliament", "legislature", "authorities"]},
    "13": {"subject": "Laws inconsistent with fundamental rights", "keywords": ["void", "derogation", "judicial review", "custom", "ordinance"]},
    "14": {"subject": "Equality before law", "keywords": ["equality", "equal protection", "discrimination", "arbitrary"]},
    "15": {"subject": "Prohibition of discrimination", "keywords": ["religion", "race", "caste", "sex", "place of birth", "women", "children"]},
    "16": {"subject": "Equality of opportunity in public employment", "keywords": ["public employment", "office", "reservation", "backward classes"]},
    "17": {"subject": "Abolition of Untouchability", "keywords": ["untouchability", "punishable", "offence"]},
    "18": {"subject": "Abolition of titles", "keywords": ["titles", "nobility", "honour", "foreign state"]},
    "19": {"subject": "Six fundamental freedoms", "keywords": ["speech", "expression", "assemble", "association", "movement", "residence", "profession"]},
    "20": {"subject": "Protection in conviction for offences", "keywords": ["ex post facto", "double jeopardy", "self-incrimination"]},
    "21": {"subject": "Protection of life and personal liberty", "keywords": ["life", "personal liberty", "procedure established by law"]},
    "21A": {"subject": "Right to education", "keywords": ["free and compulsory education", "six to fourteen", "children"]},
    "22": {"subject": "Protection against arrest and detention", "keywords": ["arrest", "twenty-four hours", "magistrate", "preventive detention"]},
    "23": {"subject": "Prohibition of human trafficking and forced labour", "keywords": ["traffic in human beings", "begar", "forced labour"]},
    "24": {"subject": "Prohibition of child labour in factories", "keywords": ["fourteen years", "factory", "mine", "hazardous"]},
    "25": {"subject": "Freedom of conscience and religion", "keywords": ["profess", "practise", "propagate", "religion", "public order"]},
    "26": {"subject": "Freedom to manage religious affairs", "keywords": ["religious denomination", "charitable", "religious affairs"]},
    "32": {"subject": "Remedies for enforcement of fundamental rights", "keywords": ["remedies", "directly move", "move the supreme court", "writs", "habeas corpus", "mandamus", "prohibition", "quo warranto", "certiorari", "enforcement of fundamental rights"]},
    "40": {"subject": "Organisation of village panchayats", "keywords": ["village panchayats", "local government"]},
    "44": {"subject": "Uniform Civil Code", "keywords": ["uniform civil code", "citizens", "throughout india"]},
    "50": {"subject": "Separation of judiciary from executive", "keywords": ["separation of judiciary", "public services"]},
    "51A": {"subject": "Fundamental Duties", "keywords": ["fundamental duties", "duty of every citizen", "national flag", "heritage"]},
    "52": {"subject": "The President of India", "keywords": ["president of india"]},
    "61": {"subject": "Procedure for impeachment of President", "keywords": ["impeachment", "violation of constitution", "two-thirds"]},
    "72": {"subject": "Pardoning power of the President", "keywords": ["pardon", "reprieve", "respite", "remission", "death sentence"]},
    "74": {"subject": "Council of Ministers to aid and advise President", "keywords": ["council of ministers", "aid and advise", "prime minister"]},
    "76": {"subject": "Attorney-General for India", "keywords": ["attorney-general", "legal advice", "audience in all courts"]},
    "123": {"subject": "Ordinance making power of President", "keywords": ["ordinance", "recess of parliament", "force of act"]},
    "124": {"subject": "Establishment of Supreme Court", "keywords": ["establishment of supreme court", "constitution of supreme court", "chief justice of india", "appointment of judges"]},
    "141": {"subject": "Law declared by Supreme Court binding", "keywords": ["binding on all courts", "law declared"]},
    "143": {"subject": "Power of President to consult Supreme Court", "keywords": ["consult supreme court", "advisory jurisdiction", "reference"]},
    "214": {"subject": "High Courts for States", "keywords": ["high court for each state"]},
    "226": {"subject": "Power of High Courts to issue writs", "keywords": ["high court", "writs", "directions", "orders", "fundamental rights"]},
    "352": {"subject": "Proclamation of National Emergency", "keywords": ["emergency", "war", "external aggression", "armed rebellion"]},
    "356": {"subject": "Failure of constitutional machinery (President's Rule)", "keywords": ["failure of constitutional machinery", "governor", "president's rule"]},
    "360": {"subject": "Financial Emergency", "keywords": ["financial emergency", "financial stability", "credit"]},
    "368": {"subject": "Power and procedure of Parliament to amend Constitution", "keywords": ["amendment", "procedure", "special majority", "ratification"]},
    "370": {"subject": "Temporary provisions for Jammu and Kashmir", "keywords": ["jammu and kashmir", "temporary"]},
}

# Known False Premise Patterns
KNOWN_FALSE_PREMISES = [
    {
        "pattern": r"(article\s+14.*(grant|confer|allow|permit|hereditary).*titles|(grant|confer|allow|permit|hereditary).*titles.*article\s+14|article\s+14.*titles\s+of\s+nobility)",
        "article": "14",
        "refutation": "Article 14 guarantees equality before the law and equal protection of the laws; it does NOT grant hereditary titles. In fact, titles are expressly abolished under Article 18 of the Constitution of India.",
        "correct_article": "18",
    },
    {
        "pattern": r"(article\s+21.*(pardon|reprieve|remit)|(pardon|reprieve|remit).*article\s+21)",
        "article": "21",
        "refutation": "Article 21 guarantees protection of life and personal liberty. The power of the President to grant pardons and remissions is provided under Article 72, not Article 21.",
        "correct_article": "72",
    },
    {
        "pattern": r"article\s+19.*(absolute\s+right\s+to\s+(defame|provoke|incite|hate)|allow.*defamation|permit.*defamation)",
        "article": "19",
        "refutation": "Article 19 does NOT guarantee an absolute right to defame or incite violence. All freedoms under Article 19(1) are subject to reasonable restrictions under clauses (2) through (6) on grounds of sovereignty, public order, and morality.",
        "correct_article": "19",
    },
    {
        "pattern": r"(article\s+1.*(secede|break\s+away|leave\s+union)|(secede|break\s+away).*article\s+1)",
        "article": "1",
        "refutation": "Article 1 declares that 'India, that is Bharat, shall be a Union of States'. The Indian Union is indestructible and States have no constitutional right to secede.",
        "correct_article": "1",
    },
    {
        "pattern": r"(article\s+51a.*(enforceable\s+by\s+(direct\s+)?writs?|court\s+enforceable)|enforceable.*article\s+51a)",
        "article": "51A",
        "refutation": "Article 51A sets out Fundamental Duties of citizens; they are moral civic obligations and are not directly enforceable by constitutional writ under Article 32 without enabling legislation.",
        "correct_article": "51A",
    },
    {
        "pattern": r"((declare|impose).*national\s+emergency.*article\s+356|article\s+356.*national\s+emergency)",
        "article": "356",
        "refutation": "Article 356 provides for President's Rule in case of failure of constitutional machinery in States. National Emergency (due to war, external aggression, or armed rebellion) is declared under Article 352.",
        "correct_article": "352",
    },
    {
        "pattern": r"(article\s+32.*empowers?\s+high\s+courts?|high\s+courts?.*article\s+32)",
        "article": "32",
        "refutation": "Article 32 empowers the Supreme Court to issue writs for Fundamental Rights. High Courts are empowered to issue writs under Article 226.",
        "correct_article": "226",
    },
    {
        "pattern": r"(article\s+44.*fundamental\s+right|fundamental\s+right.*article\s+44)",
        "article": "44",
        "refutation": "Article 44 (Uniform Civil Code) is a Directive Principle of State Policy under Part IV. Under Article 37, Directive Principles are fundamental in governance but are expressly non-justiciable and not enforceable by any court.",
        "correct_article": "44",
    },
    {
        "pattern": r"(article\s+24.*(employ.*children|mines|hazardous)|(employ.*children|mines).*article\s+24)",
        "article": "24",
        "refutation": "Article 24 contains an absolute prohibition against employing any child below the age of fourteen years in any factory, mine, or other hazardous employment.",
        "correct_article": "24",
    },
    {
        "pattern": r"(article\s+20.*(retrospective|retroactive|ex\s+post\s+facto)|(retrospective|retroactive).*article\s+20)",
        "article": "20",
        "refutation": "Article 20(1) explicitly prohibits ex post facto criminal laws; no person can be convicted of any offence except for violation of a law in force at the time of the commission of the act.",
        "correct_article": "20",
    },
]


@dataclass
class AdversarialAnalysisResult:
    """Structured outcome of query verification & classification."""
    query_type: str
    is_false_premise: bool = False
    refutation: str = ""
    target_articles: List[str] = field(default_factory=list)
    subqueries: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_prompt: str = ""
    is_out_of_domain: bool = False
    is_temporal: bool = False
    confidence: float = 1.0
    audit_notes: List[str] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        return self.needs_clarification

    @property
    def is_answerable_from_text(self) -> bool:
        return not (self.is_false_premise or self.needs_clarification or self.is_out_of_domain or self.is_temporal)


class AdversarialQueryClassifier:
    """Object-oriented wrapper for adversarial query analysis."""
    def analyze(self, query: str, history: Sequence[Any] = ()) -> AdversarialAnalysisResult:
        return analyze_adversarial_query(query, history)


_GLOBAL_ADVERSARIAL_CLASSIFIER: Optional[AdversarialQueryClassifier] = None

def get_adversarial_query_classifier() -> AdversarialQueryClassifier:
    global _GLOBAL_ADVERSARIAL_CLASSIFIER
    if _GLOBAL_ADVERSARIAL_CLASSIFIER is None:
        _GLOBAL_ADVERSARIAL_CLASSIFIER = AdversarialQueryClassifier()
    return _GLOBAL_ADVERSARIAL_CLASSIFIER


def analyze_adversarial_query(query: str, history: Sequence[Any] = ()) -> AdversarialAnalysisResult:
    """Analyze query for false premises, multi-hop dependencies, ambiguities, and out-of-domain signals."""
    q_clean = " ".join(query.strip().split())
    q_lower = q_clean.lower()
    audit: List[str] = []

    # 1. False Premise Check
    for fp in KNOWN_FALSE_PREMISES:
        if re.search(fp["pattern"], q_lower):
            audit.append(f"False premise detected matching pattern: {fp['pattern']}")
            return AdversarialAnalysisResult(
                query_type=QUERY_TYPE_FALSE_PREMISE,
                is_false_premise=True,
                refutation=fp["refutation"],
                target_articles=[fp["article"], fp.get("correct_article", fp["article"])],
                confidence=1.0,
                audit_notes=audit,
            )

    # 2. Check Article Misattribution vs. Landmark Subject
    # e.g., query explicitly pairs an article with a completely conflicting concept
    extracted_arts = re.findall(r"\b(?:article|art\.?)\s*(\d+[a-z]?)\b", q_lower)
    if len(extracted_arts) == 1:
        art_num = extracted_arts[0].upper()
        if art_num in LANDMARK_SUBJECTS:
            # Check if query asks for something directly belonging to another landmark article
            if art_num == "14" and any(k in q_lower for k in ["pardon", "death sentence", "remission"]):
                refutation = "Article 14 provides for equality before the law. The power to grant pardons and remissions belongs to the President under Article 72, not Article 14."
                return AdversarialAnalysisResult(
                    query_type=QUERY_TYPE_FALSE_PREMISE,
                    is_false_premise=True,
                    refutation=refutation,
                    target_articles=["14", "72"],
                    confidence=1.0,
                    audit_notes=["Article misattribution: pardon power paired with Article 14"],
                )
            if art_num == "21" and any(k in q_lower for k in ["ordinance", "promulgate ordinance"]):
                refutation = "Article 21 protects life and personal liberty. Ordinance-making powers belong to the President under Article 123 and Governors under Article 213, not Article 21."
                return AdversarialAnalysisResult(
                    query_type=QUERY_TYPE_FALSE_PREMISE,
                    is_false_premise=True,
                    refutation=refutation,
                    target_articles=["21", "123"],
                    confidence=1.0,
                    audit_notes=["Article misattribution: ordinance power paired with Article 21"],
                )

    # 3. Temporal / Dynamic Inquiries (Non-corpus events)
    from app.retrieval.evidence_guard import is_temporal_or_dynamic_query
    is_temp, temp_msg = is_temporal_or_dynamic_query(q_clean)
    if is_temp:
        audit.append("Temporal/dynamic inquiry detected")
        return AdversarialAnalysisResult(
            query_type=QUERY_TYPE_TEMPORAL_DYNAMIC,
            is_temporal=True,
            refutation=temp_msg or "The Constitution text does not record dynamic current events.",
            confidence=1.0,
            audit_notes=audit,
        )

    # 4. Out of Domain check
    from app.retrieval.evidence_guard import evaluate_query_domain
    in_domain, ood_msg = evaluate_query_domain(q_clean)
    if not in_domain:
        audit.append(f"Domain rejection: {ood_msg}")
        return AdversarialAnalysisResult(
            query_type=QUERY_TYPE_OUT_OF_DOMAIN,
            is_out_of_domain=True,
            refutation=ood_msg,
            confidence=1.0,
            audit_notes=audit,
        )

    # 5. Ambiguous / Underspecified Queries
    ambiguous_patterns = [
        (r"^(what|who)\s+is\s+the\s+quota\b", "Could you please specify whether you are asking about reservation in public employment (Article 16), educational admissions (Article 15), or legislative seats (Articles 330/332)?"),
        (r"^who\s+(can\s+)?issues?\s+ordinances?\??$", "Could you please clarify whether you are inquiring about the ordinance-making power of the Union President (Article 123) or State Governors (Article 213)?"),
        (r"^how\s+to\s+amend\??$", "Could you please specify which part of the Constitution you wish to amend, or whether you are inquiring about the general amendment procedure under Article 368?"),
        (r"^what\s+are\s+the\s+writs\??$", "The Constitution provides writ jurisdiction under Article 32 (Supreme Court) and Article 226 (High Courts). Would you like an overview of both or a specific jurisdiction?"),
    ]
    for pat, prompt_msg in ambiguous_patterns:
        if re.search(pat, q_lower):
            audit.append(f"Ambiguous query matched pattern: {pat}")
            return AdversarialAnalysisResult(
                query_type=QUERY_TYPE_AMBIGUOUS,
                needs_clarification=True,
                clarification_prompt=prompt_msg,
                confidence=0.95,
                audit_notes=audit,
            )

    # 6. Multi-Hop / Multi-Article Reasoning Detection
    multi_art_matches = re.findall(r"\b(?:article|art\.?)\s*(\d+[a-z]?)\b", q_lower)
    distinct_arts = list(dict.fromkeys([a.upper() for a in multi_art_matches]))
    has_comparison = any(w in q_lower for w in ["compare", "comparison", "difference", "versus", "vs", "relationship between", "combined effect"])

    if len(distinct_arts) >= 2 or has_comparison or "golden triangle" in q_lower:
        target_arts = distinct_arts
        if "golden triangle" in q_lower and not target_arts:
            target_arts = ["14", "19", "21"]
        subqueries = [f"What does Article {a} provide in the Constitution of India?" for a in target_arts]
        audit.append(f"Multi-hop reasoning detected for articles: {target_arts}")
        return AdversarialAnalysisResult(
            query_type=QUERY_TYPE_MULTI_HOP,
            target_articles=target_arts,
            subqueries=subqueries,
            confidence=0.95,
            audit_notes=audit,
        )

    # 7. Exact Clause / Subclause Lookup Detection
    clause_match = re.search(r"\b(?:article|art\.?)\s*(\d+[a-z]?)\s*(?:clause|\()\s*(\d+|[a-z])", q_lower)
    if clause_match:
        art_no = clause_match.group(1).upper()
        audit.append(f"Clause-level extraction detected for Article {art_no}")
        return AdversarialAnalysisResult(
            query_type=QUERY_TYPE_CLAUSE_LOOKUP,
            target_articles=[art_no],
            confidence=0.95,
            audit_notes=audit,
        )

    semantic_targets = []
    if "golden triangle" in q_lower:
        semantic_targets.extend(["14", "19", "21"])
    if "discriminate" in q_lower or "discrimination" in q_lower:
        semantic_targets.extend(["14", "15"])
    if "education" in q_lower and ("children" in q_lower or "compulsory" in q_lower or "six" in q_lower or "14" in q_lower or "fourteen" in q_lower):
        semantic_targets.append("21A")

    for art, info in LANDMARK_SUBJECTS.items():
        if any(kw in q_lower for kw in info.get("keywords", [])):
            if art not in semantic_targets:
                semantic_targets.append(art)

    all_targets = list(dict.fromkeys(distinct_arts + semantic_targets))

    if all_targets:
        return AdversarialAnalysisResult(
            query_type=QUERY_TYPE_MULTI_HOP if len(all_targets) > 1 else QUERY_TYPE_EXACT_ARTICLE,
            target_articles=all_targets,
            confidence=0.88,
            audit_notes=audit,
        )

    return AdversarialAnalysisResult(
        query_type=QUERY_TYPE_GENERAL_SEMANTIC,
        confidence=0.80,
        audit_notes=audit,
    )
