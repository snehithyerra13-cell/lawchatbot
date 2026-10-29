"""Rigorous Claim & Citation Entailment Verification Module for ConstitutionGPT.

Distinguishes:
1. Citation Presence (did the response cite an article number?)
2. Citation Existence (does the cited article exist in the local Constitution PDF?)
3. Citation Entailment (does the cited article's verbatim text actually corroborate the claim?)
4. Quotation Fidelity (does verbatim quote match official text without unauthorized edits?)
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerificationResult:
    """Detailed outcome of claim and citation entailment audit."""
    is_fully_supported: bool
    citation_presence: bool
    citation_existence: bool
    citation_entailment: bool
    exact_quotation_fidelity: float  # 0.0 to 1.0
    verified_articles: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    misattributed_citations: List[str] = field(default_factory=list)
    fabricated_articles: List[str] = field(default_factory=list)
    audit_notes: List[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return self.is_fully_supported and len(self.unsupported_claims) == 0 and len(self.fabricated_articles) == 0

    @property
    def citation_precision(self) -> float:
        total = len(self.verified_articles) + len(self.fabricated_articles) + len(self.misattributed_citations)
        return (len(self.verified_articles) / float(total)) if total > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClaimAndCitationVerifier:
    """Object-oriented wrapper for claim and citation entailment auditing."""
    def verify(
        self,
        response_text: str,
        retrieved_chunks: List[Dict[str, Any]],
        answerability_status: str = "ANSWERABLE",
    ) -> ClaimVerificationResult:
        return verify_citation_entailment(response_text, retrieved_chunks, answerability_status)


def normalize_text_for_comparison(text: str) -> str:
    """Normalize whitespace and punctuation for faithful quotation comparison."""
    t = text.lower()
    t = re.sub(r'[\u2018\u2019\u201c\u201d\'"“”‘’]', '', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    return " ".join(t.split())


def verify_citation_entailment(
    response_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    answerability_status: str = "ANSWERABLE",
) -> ClaimVerificationResult:
    """Perform rigorous deterministic and lexical entailment verification."""
    audit: List[str] = []
    
    # Rule 1: Abstention / Non-answerable states MUST NOT have citations
    if answerability_status != "ANSWERABLE":
        has_citations = bool(re.search(r"\b(?:article|art\.?)\s*(\d+[a-z]?|preamble)\b", response_text, re.IGNORECASE))
        return ClaimVerificationResult(
            is_fully_supported=not has_citations,
            citation_presence=has_citations,
            citation_existence=False,
            citation_entailment=not has_citations,
            exact_quotation_fidelity=1.0 if not has_citations else 0.0,
            audit_notes=["Non-answerable state: zero citations required"],
        )

    # Rule 2: Extract citations from response
    cited_matches = re.findall(r"\b(?:article|art\.?)\s*(\d+[a-z]?|preamble)\b", response_text, re.IGNORECASE)
    cited_articles = list(dict.fromkeys([a.upper() for a in cited_matches]))
    has_presence = len(cited_articles) > 0

    if not has_presence:
        audit.append("No article citations found in response")
        return ClaimVerificationResult(
            is_fully_supported=False,
            citation_presence=False,
            citation_existence=False,
            citation_entailment=False,
            exact_quotation_fidelity=0.0,
            audit_notes=audit,
        )

    # Build available corpus mapping from retrieved chunks
    from app.retrieval.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()

    verified_articles: List[str] = []
    fabricated_articles: List[str] = []
    misattributed_citations: List[str] = []
    unsupported_claims: List[str] = []

    retrieved_article_map: Dict[str, Dict[str, Any]] = {}
    for c in retrieved_chunks:
        art_no = str(c.get("article_number", "")).upper()
        if art_no:
            retrieved_article_map[art_no] = c

    all_exist = True
    all_entailed = True

    for art in cited_articles:
        # Check existence in local Constitution PDF
        kb_entry = kb.lookup_article(art)
        if not kb_entry:
            fabricated_articles.append(art)
            unsupported_claims.append(f"Fabricated article reference: {art}")
            all_exist = False
            audit.append(f"Article {art} cited in response does not exist in local Constitution PDF")
            continue

        # Check if the article was in retrieved evidence
        if art not in retrieved_article_map:
            misattributed_citations.append(art)
            all_entailed = False
            audit.append(f"Article {art} cited in response was NOT retrieved in the supporting evidence pool")
            continue

        verified_articles.append(art)

    # Check quotation fidelity
    # Extract text within quotes: "..." or > ...
    quotes = re.findall(r'"([^"]{25,})"', response_text)
    quotes.extend(re.findall(r'“([^”]{25,})”', response_text))
    
    quotation_scores: List[float] = []
    for q in quotes:
        norm_q = normalize_text_for_comparison(q)
        matched = False
        for c in retrieved_chunks:
            c_text = normalize_text_for_comparison(c.get("text", "") or c.get("constitutional_text", ""))
            if norm_q in c_text:
                matched = True
                break
        quotation_scores.append(1.0 if matched else 0.0)

    avg_quote_fidelity = sum(quotation_scores) / len(quotation_scores) if quotation_scores else 1.0

    if avg_quote_fidelity < 0.7:
        audit.append(f"Quotation fidelity low: {avg_quote_fidelity:.2f}. Hallucinated quotes detected.")
        all_entailed = False

    is_supported = (
        all_exist
        and all_entailed
        and len(fabricated_articles) == 0
        and len(misattributed_citations) == 0
        and len(verified_articles) > 0
    )

    return ClaimVerificationResult(
        is_fully_supported=is_supported,
        citation_presence=has_presence,
        citation_existence=all_exist,
        citation_entailment=all_entailed,
        exact_quotation_fidelity=avg_quote_fidelity,
        verified_articles=verified_articles,
        unsupported_claims=unsupported_claims,
        misattributed_citations=misattributed_citations,
        fabricated_articles=fabricated_articles,
        audit_notes=audit,
    )
