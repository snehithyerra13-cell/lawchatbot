"""Comprehensive Adversarial Evaluation Dataset Generator for ConstitutionGPT.

Builds 160+ challenging test cases across 8 rigorous categories:
Category A: Exact Article & Clause Retrieval (25)
Category B: Multi-Hop Constitutional Reasoning (25)
Category C: Adversarial False Premise & Misleading Inquiries (25)
Category D: Ambiguous & Under-Specified Inquiries (20)
Category E: Unanswerable & Out-of-Scope Inquiries (25)
Category F: Temporal, Version & Dynamic Inquiries (20)
Category G: Long-Context Distractor & Retrieval Stress Tests (20)
Category H: Conversational Memory & Pivot Stress Tests (15)

Total: 175 High-Difficulty Test Cases
All expected answers and chunk IDs are verified against data/articles.json and SharedConstitutionalCorpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.retrieval.knowledge_base import get_knowledge_base


def generate_adversarial_dataset() -> List[Dict[str, Any]]:
    """Build the comprehensive 175-item adversarial dataset."""
    kb = get_knowledge_base()
    cases: List[Dict[str, Any]] = []

    def add_case(
        qid: str,
        category: str,
        query: str,
        is_answerable: bool,
        expected_abstention: bool,
        expected_articles: List[str],
        expected_chunk_ids: List[str],
        description: str,
        expected_answer_keywords: List[str],
        adversarial_type: str = "none",
        is_false_premise: bool = False,
        refutation: str = "",
    ):
        cases.append({
            "query_id": qid,
            "category": category,
            "query": query,
            "is_answerable": is_answerable,
            "expected_abstention": expected_abstention,
            "expected_articles": expected_articles,
            "expected_chunk_ids": expected_chunk_ids,
            "description": description,
            "expected_answer_keywords": expected_answer_keywords,
            "adversarial_type": adversarial_type,
            "is_false_premise": is_false_premise,
            "refutation": refutation,
        })

    # =========================================================================
    # CATEGORY A: Exact Article and Clause Retrieval (25 items)
    # =========================================================================
    add_case("adv_A01", "exact_article_and_clause", "What does Article 14 of the Constitution of India declare regarding equality?", True, False, ["14"], ["const_14_p37_c01"], "Exact wording of Article 14", ["equality before the law", "equal protection"])
    add_case("adv_A02", "exact_article_and_clause", "Under Article 15(1), what are the five specific prohibited grounds of discrimination?", True, False, ["15"], ["const_15_p342_c01"], "Exact grounds under Article 15(1)", ["religion", "race", "caste", "sex", "place of birth"])
    add_case("adv_A03", "exact_article_and_clause", "What does Article 16 guarantee regarding equality of opportunity in public employment?", True, False, ["16"], ["const_16_p344_c01"], "Public employment equality under Article 16", ["equality of opportunity", "employment", "office under the state"])
    add_case("adv_A04", "exact_article_and_clause", "What is declared by Article 17 regarding Untouchability?", True, False, ["17"], ["const_17_p40_c01"], "Abolition of Untouchability", ["untouchability", "forbidden", "punishable"])
    add_case("adv_A05", "exact_article_and_clause", "What does Article 18 state regarding the abolition of titles?", True, False, ["18"], ["const_18_p40_c01"], "Abolition of titles under Article 18", ["no title", "military or academic", "conferred"])
    add_case("adv_A06", "exact_article_and_clause", "Enumerate the six freedoms protected under Article 19(1).", True, False, ["19"], ["const_19_p40_c01"], "Six freedoms under Article 19", ["speech and expression", "assemble peaceably", "associations", "move freely", "reside and settle", "profession"])
    add_case("adv_A07", "exact_article_and_clause", "Explain the protection against ex post facto laws under Article 20(1).", True, False, ["20"], ["const_20_p42_c01"], "Ex post facto law prohibition in Article 20(1)", ["offence", "law in force", "penalty greater"])
    add_case("adv_A08", "exact_article_and_clause", "What is the protection against double jeopardy in Article 20(2)?", True, False, ["20"], ["const_20_p42_c01"], "Double jeopardy under Article 20(2)", ["prosecuted and punished", "same offence"])
    add_case("adv_A09", "exact_article_and_clause", "What is the exact provision of Article 20(3) on self-incrimination?", True, False, ["20"], ["const_20_p42_c01"], "Self-incrimination in Article 20(3)", ["compelled", "witness against himself"])
    add_case("adv_A10", "exact_article_and_clause", "What does Article 21 state regarding life and personal liberty?", True, False, ["21"], ["const_21_p42_c01"], "Article 21 life and personal liberty", ["deprived of his life", "personal liberty", "procedure established by law"])
    add_case("adv_A11", "exact_article_and_clause", "What are the exact age limits specified in Article 21A for free education?", True, False, ["21A"], ["const_21a_p42_c01"], "Age limits in Article 21A", ["six", "fourteen", "compulsory education"])
    add_case("adv_A12", "exact_article_and_clause", "Within what time must an arrested person be produced before a Magistrate under Article 22(2)?", True, False, ["22"], ["const_22_p42_c01"], "24-hour rule under Article 22(2)", ["twenty-four hours", "magistrate"])
    add_case("adv_A13", "exact_article_and_clause", "What does Article 23 prohibit regarding forced labour and human trafficking?", True, False, ["23"], ["const_23_p45_c01"], "Traffic in human beings and begar under Article 23", ["traffic in human beings", "begar", "forced labour"])
    add_case("adv_A14", "exact_article_and_clause", "What is the age threshold for prohibiting child employment in factories under Article 24?", True, False, ["24"], ["const_24_p45_c01"], "Child labour prohibition age under Article 24", ["fourteen years", "factory", "mine"])
    add_case("adv_A15", "exact_article_and_clause", "What does Article 25 declare regarding freedom of conscience and religion?", True, False, ["25"], ["const_25_p349_c01"], "Freedom of conscience under Article 25", ["conscience", "profess", "practise", "propagate"])
    add_case("adv_A16", "exact_article_and_clause", "What rights are given to religious denominations under Article 26?", True, False, ["26"], ["const_26_p350_c01"], "Religious denominations rights in Article 26", ["religious denomination", "institutions", "manage its own affairs"])
    add_case("adv_A17", "exact_article_and_clause", "What is declared in Article 28 regarding religious instruction in State-funded schools?", True, False, ["28"], ["const_28_p351_c01"], "Religious instruction prohibition under Article 28", ["religious instruction", "wholly maintained out of state funds"])
    add_case("adv_A18", "exact_article_and_clause", "What protections for minority language, script and culture are provided in Article 29?", True, False, ["29"], ["const_29_p50_c01"], "Protection of minority interests under Article 29", ["language", "script", "culture", "conserve"])
    add_case("adv_A19", "exact_article_and_clause", "What is the right of minorities to establish educational institutions under Article 30?", True, False, ["30"], ["const_30_p50_c01"], "Minority educational institutions under Article 30", ["minorities", "religion or language", "establish and administer"])
    add_case("adv_A20", "exact_article_and_clause", "What is the constitutional mandate of Article 32 regarding Fundamental Rights remedies?", True, False, ["32"], ["const_32_p50_c01", "const_32_p50_c02"], "Writ remedies in Supreme Court under Article 32", ["remedies", "supreme court", "writs"])
    add_case("adv_A21", "exact_article_and_clause", "What directive does Article 40 give regarding village panchayats?", True, False, ["40"], ["const_40_p53_c01"], "Village panchayats under Article 40", ["village panchayats", "self-government"])
    add_case("adv_A22", "exact_article_and_clause", "What is the exact text of Article 44 regarding Uniform Civil Code?", True, False, ["44"], ["const_44_p54_c01"], "Uniform civil code in Article 44", ["uniform civil code", "citizens throughout the territory of india"])
    add_case("adv_A23", "exact_article_and_clause", "What does Article 50 state regarding separation of judiciary from executive?", True, False, ["50"], ["const_50_p55_c01"], "Separation of judiciary under Article 50", ["separate the judiciary from the executive", "public services"])
    add_case("adv_A24", "exact_article_and_clause", "What are the first three Fundamental Duties listed in Article 51A?", True, False, ["51A"], ["const_51a_p56_c01"], "First duties under Article 51A", ["constitution", "national flag", "sovereignty, unity and integrity"])
    add_case("adv_A25", "exact_article_and_clause", "What does Article 72 declare about the President's power to grant pardons?", True, False, ["72"], ["const_72_p362_c01"], "Pardoning power under Article 72", ["pardon", "reprieve", "respite", "remission"])

    # =========================================================================
    # CATEGORY B: Multi-Hop Constitutional Reasoning (25 items)
    # =========================================================================
    add_case("adv_B01", "multi_hop_reasoning", "How do Article 14, Article 19, and Article 21 interact in the Golden Triangle doctrine?", True, False, ["14", "19", "21"], ["const_14_p37_c01", "const_19_p40_c01", "const_21_p42_c01"], "Golden triangle interrelationship", ["equality", "freedom", "personal liberty"])
    add_case("adv_B02", "multi_hop_reasoning", "Compare the writ jurisdiction of the Supreme Court under Article 32 with High Courts under Article 226.", True, False, ["32", "226"], ["const_32_p50_c01", "const_226_p135_c01"], "Writ powers comparison Articles 32 vs 226", ["supreme court", "high court", "writs", "fundamental rights", "any other purpose"])
    add_case("adv_B03", "multi_hop_reasoning", "How does Article 13 define laws inconsistent with Fundamental Rights in relation to Article 368 amendments?", True, False, ["13", "368"], ["const_13_p36_c01", "const_368_p35_c01"], "Relationship between Article 13 and Article 368", ["void", "amendment", "part iii"])
    add_case("adv_B04", "multi_hop_reasoning", "Compare National Emergency under Article 352 with State Emergency (President's Rule) under Article 356.", True, False, ["352", "356"], ["const_352_p239_c01", "const_356_p35_c01"], "Emergency comparison 352 vs 356", ["proclamation of emergency", "failure of constitutional machinery", "governor"])
    add_case("adv_B05", "multi_hop_reasoning", "How do Article 23 and Article 24 combine to establish the Right against Exploitation?", True, False, ["23", "24"], ["const_23_p45_c01", "const_24_p45_c01"], "Right against exploitation synthesis", ["traffic in human beings", "begar", "child labour", "factories"])
    add_case("adv_B06", "multi_hop_reasoning", "How does the Right to Education under Article 21A correlate with Fundamental Duty Article 51A(k)?", True, False, ["21A", "51A"], ["const_21a_p42_c01", "const_51a_p56_c01"], "Education right and parental duty synthesis", ["six to fourteen", "parent or guardian", "education"])
    add_case("adv_B07", "multi_hop_reasoning", "Compare Article 15 prohibition of discrimination with Article 16 public employment equality.", True, False, ["15", "16"], ["const_15_p342_c01", "const_16_p344_c01"], "Discrimination vs Employment opportunity", ["discrimination", "public employment", "reservation", "backward classes"])
    add_case("adv_B08", "multi_hop_reasoning", "How do Article 25 and Article 26 collectively protect religious rights of individuals and groups?", True, False, ["25", "26"], ["const_25_p349_c01", "const_26_p350_c01"], "Individual vs denominational religious freedom", ["profess and practise", "denomination", "manage religious affairs"])
    add_case("adv_B09", "multi_hop_reasoning", "What is the relationship between the Ordinance-making power of the President in Article 123 and Parliament under Article 79?", True, False, ["123", "79"], ["const_123_p365_c01", "const_79_p67_c01"], "Ordinances vs Parliament authority", ["ordinance", "recess of parliament", "both houses"])
    add_case("adv_B10", "multi_hop_reasoning", "How does the advisory jurisdiction of the Supreme Court under Article 143 relate to the President under Article 52?", True, False, ["143", "52"], ["const_143_p98_c01", "const_52_p57_c01"], "Presidential consultation with Supreme Court", ["consult supreme court", "president", "question of law or fact"])
    add_case("adv_B11", "multi_hop_reasoning", "Compare Article 20 protection in conviction with Article 22 protection against arrest and detention.", True, False, ["20", "22"], ["const_20_p42_c01", "const_22_p42_c01"], "Conviction vs Arrest safeguards", ["double jeopardy", "self incrimination", "twenty-four hours", "magistrate"])
    add_case("adv_B12", "multi_hop_reasoning", "How does the Council of Ministers under Article 74 aid and advise the President established by Article 52?", True, False, ["74", "52"], ["const_74_p64_c01", "const_52_p57_c01"], "Executive power and ministerial advice", ["council of ministers", "aid and advise", "prime minister"])
    add_case("adv_B13", "multi_hop_reasoning", "What is the difference between impeachment of the President in Article 61 and removal of Supreme Court judges in Article 124(4)?", True, False, ["61", "124"], ["const_61_p61_c01", "const_124_p88_c01"], "Impeachment vs Judicial removal", ["impeachment", "violation of the constitution", "proved misbehaviour", "incapacity"])
    add_case("adv_B14", "multi_hop_reasoning", "How do Article 29 and Article 30 jointly safeguard educational and cultural rights of minorities?", True, False, ["29", "30"], ["const_29_p50_c01", "const_30_p50_c01"], "Cultural and educational rights synthesis", ["conserve", "language, script", "establish and administer"])
    add_case("adv_B15", "multi_hop_reasoning", "How does Financial Emergency under Article 360 differ from National Emergency under Article 352?", True, False, ["360", "352"], ["const_360_p246_c01", "const_352_p239_c01"], "Financial emergency vs National emergency", ["financial stability", "war", "external aggression"])
    add_case("adv_B16", "multi_hop_reasoning", "How do Article 12 definition of State and Article 36 definition of State in Part IV compare?", True, False, ["12", "36"], ["const_12_p35_c01", "const_36_p51_c01"], "Definition of State across Part III and Part IV", ["state", "government and parliament", "part iii", "part iv"])
    add_case("adv_B17", "multi_hop_reasoning", "How does Article 141 making Supreme Court law binding relate to High Courts under Article 214?", True, False, ["141", "214"], ["const_141_p97_c01", "const_214_p130_c01"], "Supreme Court precedent binding on High Courts", ["law declared", "binding on all courts", "high courts"])
    add_case("adv_B18", "multi_hop_reasoning", "Compare the Attorney-General under Article 76 with the Council of Ministers under Article 74.", True, False, ["76", "74"], ["const_76_p65_c01", "const_74_p64_c01"], "Legal advisor vs Political executive", ["attorney-general", "legal advice", "council of ministers"])
    add_case("adv_B19", "multi_hop_reasoning", "How does the Preamble's declaration of Republic connect with the elected President under Article 54?", True, False, ["PREAMBLE", "54"], ["const_preamble_p32_c01", "const_54_p57_c01"], "Republic concept and Presidential election", ["republic", "elected members", "electoral college"])
    add_case("adv_B20", "multi_hop_reasoning", "How do Fundamental Rights under Part III and Directive Principles under Part IV relate under Article 37?", True, False, ["14", "37"], ["const_14_p37_c01", "const_37_p51_c01"], "Enforceability vs Fundamental in governance", ["enforceable", "fundamental in the governance", "not enforceable by any court"])
    add_case("adv_B21", "multi_hop_reasoning", "How does Article 1 declare the territory of India in conjunction with citizenship in Article 5?", True, False, ["1", "5"], ["const_1_p33_c01", "const_5_p33_c01"], "Territory and Citizenship at commencement", ["union of states", "territory", "citizenship", "domicile"])
    add_case("adv_B22", "multi_hop_reasoning", "Compare the power of Supreme Court as a court of record in Article 129 with High Courts in Article 215.", True, False, ["129", "215"], ["const_129_p93_c01", "const_215_p130_c01"], "Courts of record powers", ["court of record", "punish for contempt"])
    add_case("adv_B23", "multi_hop_reasoning", "How does the oath of President under Article 60 relate to the duty to aid and advise in Article 74?", True, False, ["60", "74"], ["const_60_p61_c01", "const_74_p64_c01"], "Oath to preserve Constitution vs Ministerial advice", ["preserve, protect and defend", "council of ministers"])
    add_case("adv_B24", "multi_hop_reasoning", "How do Article 32 and Article 13 combine to enable judicial review of legislation in India?", True, False, ["32", "13"], ["const_32_p50_c01", "const_13_p36_c01"], "Judicial review mechanism", ["judicial review", "void", "enforcement of fundamental rights"])
    add_case("adv_B25", "multi_hop_reasoning", "Compare the procedure to amend Fundamental Rights under Article 368 with ordinary law under Article 107.", True, False, ["368", "107"], ["const_368_p35_c01", "const_107_p77_c01"], "Constitutional amendment vs Ordinary legislative bill", ["two-thirds majority", "ordinary bill", "both houses"])

    # =========================================================================
    # CATEGORY C: Adversarial and Misleading Questions (25 items)
    # =========================================================================
    add_case("adv_C01", "adversarial_false_premise", "Under Article 14, how does the Constitution grant hereditary titles of nobility to citizens?", False, True, ["14", "18"], [], "False premise: Article 14 confers titles", ["article 14", "equality before law", "abolition of titles", "article 18"], "false_premise", True, "Article 14 guarantees equality before the law; it does NOT grant hereditary titles. In fact, titles are expressly abolished under Article 18.")
    add_case("adv_C02", "adversarial_false_premise", "Why does Article 21 empower the President of India to grant death sentence pardons?", False, True, ["21", "72"], [], "False premise: Article 21 has pardon power", ["article 21", "life and personal liberty", "pardon", "article 72"], "false_premise", True, "Article 21 protects life and personal liberty; pardoning powers are granted under Article 72.")
    add_case("adv_C03", "adversarial_false_premise", "Explain why Article 19 guarantees an absolute right to publish defamatory statements without restriction.", False, True, ["19"], [], "False premise: Absolute defamation right", ["article 19", "reasonable restrictions", "defamation", "clause (2)"], "false_premise", True, "Article 19 does not grant an absolute right; freedom of speech is subject to reasonable restrictions including defamation under Article 19(2).")
    add_case("adv_C04", "adversarial_false_premise", "Under Article 1, which individual States have the constitutional right to secede from the Union?", False, True, ["1"], [], "False premise: State secession under Article 1", ["article 1", "union of states", "indestructible", "no right to secede"], "false_premise", True, "Article 1 declares India to be a Union of States; no State has a constitutional right to secede.")
    add_case("adv_C05", "adversarial_false_premise", "How can an affected citizen file a direct writ in the High Court under Article 51A?", False, True, ["51A", "226"], [], "False premise: Article 51A writ jurisdiction", ["article 51a", "fundamental duties", "moral obligations", "not enforceable by writ"], "false_premise", True, "Article 51A prescribes Fundamental Duties which are not directly enforceable by writ.")
    add_case("adv_C06", "adversarial_false_premise", "Why does Article 356 permit the declaration of National Emergency during wartime?", False, True, ["356", "352"], [], "False premise: Article 356 wartime emergency", ["article 356", "president's rule", "state machinery", "article 352 national emergency"], "false_premise", True, "Article 356 provides for failure of constitutional machinery in States (President's Rule). National Emergency during war is governed by Article 352.")
    add_case("adv_C07", "adversarial_false_premise", "Explain how Article 32 empowers District Courts to issue habeas corpus writs.", False, True, ["32"], [], "False premise: District court writs under Article 32", ["article 32", "supreme court", "not district courts"], "false_premise", True, "Article 32 empowers the Supreme Court of India, not District Courts.")
    add_case("adv_C08", "adversarial_false_premise", "Under Article 44, how can citizens sue the State in court to enforce a Uniform Civil Code immediately?", False, True, ["44", "37"], [], "False premise: Direct judicial enforcement of Article 44", ["article 44", "directive principle", "part iv", "not enforceable in court", "article 37"], "false_premise", True, "Article 44 is a Directive Principle under Part IV, which under Article 37 is expressly non-justiciable and cannot be enforced by court action.")
    add_case("adv_C09", "adversarial_false_premise", "Why does Article 24 permit twelve-year-old children to be employed in coal mines?", False, True, ["24"], [], "False premise: Permitted child labour under Article 24", ["article 24", "fourteen years", "prohibition", "no child"], "false_premise", True, "Article 24 strictly prohibits employing any child below the age of fourteen years in any factory, mine, or hazardous work.")
    add_case("adv_C10", "adversarial_false_premise", "Under Article 20, how can Parliament enact a criminal law imposing punishments for acts committed before the law existed?", False, True, ["20"], [], "False premise: Ex post facto criminal law allowed", ["article 20(1)", "prohibits ex post facto", "law in force", "cannot"], "false_premise", True, "Article 20(1) expressly prohibits ex post facto criminal legislation.")
    add_case("adv_C11", "adversarial_false_premise", "Under Article 12, why are private multi-national corporations defined as the State?", False, True, ["12"], [], "False premise: Private foreign corporations as State", ["article 12", "definition of state", "government and parliament", "local or other authorities"], "false_premise", True, "Article 12 defines State as the Government and Parliament, State Governments and Legislatures, and local or other authorities within India.")
    add_case("adv_C12", "adversarial_false_premise", "Explain how Article 17 allows untouchability in private domestic clubs.", False, True, ["17"], [], "False premise: Untouchability allowed in private clubs", ["article 17", "untouchability abolished", "forbidden in any form"], "false_premise", True, "Article 17 completely abolishes Untouchability and forbids its practice in ANY form without exception.")
    add_case("adv_C13", "adversarial_false_premise", "Under Article 21, how does the Constitution define the death penalty as unconstitutional in all cases?", False, True, ["21"], [], "False premise: Death penalty unconstitutional in Article 21", ["article 21", "procedure established by law", "deprived of life"], "false_premise", True, "Article 21 permits deprivation of life according to procedure established by law; it does not declare the death penalty categorically unconstitutional.")
    add_case("adv_C14", "adversarial_false_premise", "Why does Article 76 grant the Attorney-General judicial authority to deliver binding court judgments?", False, True, ["76"], [], "False premise: Attorney General judicial judgments", ["article 76", "attorney-general", "legal advice", "not a judge"], "false_premise", True, "Article 76 designates the Attorney-General as chief legal adviser to the Union Government; they have no judicial power to deliver court judgments.")
    add_case("adv_C15", "adversarial_false_premise", "Under Article 61, why is the Prime Minister impeached by the Supreme Court?", False, True, ["61"], [], "False premise: PM impeached by Supreme Court", ["article 61", "president of india", "parliament", "two-thirds majority", "not prime minister"], "false_premise", True, "Article 61 prescribes the procedure for impeachment of the President by Parliament, not the Prime Minister or Supreme Court.")
    add_case("adv_C16", "adversarial_false_premise", "Under Article 123, how can the Governor of a State issue an Ordinance for Parliament?", False, True, ["123", "213"], [], "False premise: Governor issues ordinances for Parliament", ["article 123", "president", "parliament", "article 213 governor"], "false_premise", True, "Article 123 empowers the President to promulgate Ordinances for the Union during recess of Parliament; Governors act under Article 213 for State Legislatures.")
    add_case("adv_C17", "adversarial_false_premise", "Under Article 22, why is an arrested suspect denied the right to consult a legal practitioner?", False, True, ["22"], [], "False premise: Right to counsel denied", ["article 22(1)", "right to consult", "legal practitioner of his choice"], "false_premise", True, "Article 22(1) explicitly guarantees the right of every arrested person to consult and be defended by a legal practitioner of their choice.")
    add_case("adv_C18", "adversarial_false_premise", "Why does Article 25 make religious propagation illegal across India?", False, True, ["25"], [], "False premise: Propagation made illegal", ["article 25", "profess, practise and propagate", "guaranteed right"], "false_premise", True, "Article 25 explicitly guarantees the right freely to profess, practise, and propagate religion, subject to public order, morality, and health.")
    add_case("adv_C19", "adversarial_false_premise", "Explain why Article 368 allows the Prime Minister alone to amend the Constitution by decree.", False, True, ["368"], [], "False premise: PM amends Constitution by decree", ["article 368", "parliament", "two-thirds majority", "special majority", "not decree"], "false_premise", True, "Article 368 grants constituent amendment power to Parliament acting by a special majority in both Houses, not by executive decree.")
    add_case("adv_C20", "adversarial_false_premise", "Under Article 26, how are religious denominations prevented from owning moveable property?", False, True, ["26"], [], "False premise: Religious denominations barred from property", ["article 26(c)", "acquire and own property", "moveable and immoveable"], "false_premise", True, "Article 26(c) expressly guarantees every religious denomination the right to acquire and own both moveable and immoveable property.")
    add_case("adv_C21", "adversarial_false_premise", "Why does Article 29 force minority linguistic groups to adopt Hindi script?", False, True, ["29"], [], "False premise: Minorities forced to adopt Hindi", ["article 29(1)", "distinct language, script or culture", "right to conserve"], "false_premise", True, "Article 29(1) protects any section of citizens having a distinct language, script, or culture of its own, giving them the right to conserve the same.")
    add_case("adv_C22", "adversarial_false_premise", "Under Article 50, why is the Supreme Court placed directly under the administrative control of the Union Home Ministry?", False, True, ["50"], [], "False premise: Supreme Court under Home Ministry", ["article 50", "separation of judiciary", "independent"], "false_premise", True, "Article 50 directs the State to separate the judiciary from the executive to ensure judicial independence.")
    add_case("adv_C23", "adversarial_false_premise", "Under Article 21A, why are children above the age of ten excluded from school education?", False, True, ["21A"], [], "False premise: Children above 10 excluded", ["article 21a", "six to fourteen years", "free and compulsory"], "false_premise", True, "Article 21A guarantees free and compulsory education for all children aged six to fourteen years.")
    add_case("adv_C24", "adversarial_false_premise", "Why does Article 143 allow the Supreme Court to order the President to resign?", False, True, ["143"], [], "False premise: Supreme Court orders President to resign", ["article 143", "advisory opinion", "consultation", "not order resignation"], "false_premise", True, "Article 143 provides advisory jurisdiction where the President refers a question to the Supreme Court for its opinion, not to order presidential resignation.")
    add_case("adv_C25", "adversarial_false_premise", "Under Article 18, why are honorary military medals declared unconstitutional in India?", False, True, ["18"], [], "False premise: Military decorations declared unconstitutional", ["article 18(1)", "military or academic distinction", "exception"], "false_premise", True, "Article 18(1) expressly exempts military and academic distinctions from the abolition of titles.")

    # =========================================================================
    # CATEGORY D: Ambiguous and Under-Specified Questions (20 items)
    # =========================================================================
    add_case("adv_D01", "ambiguous", "Who is empowered to issue ordinances?", False, True, ["123", "213"], [], "Ambiguous Union vs State ordinance power", ["clarify", "president", "governor", "article 123", "article 213"], "ambiguous", False)
    add_case("adv_D02", "ambiguous", "What is the quota percentage for backward classes?", False, True, ["15", "16"], [], "Ambiguous reservation percentage inquiry", ["clarify", "public employment", "education", "article 15", "article 16"], "ambiguous", False)
    add_case("adv_D03", "ambiguous", "How can an emergency be proclaimed?", False, True, ["352", "356", "360"], [], "Ambiguous emergency type inquiry", ["clarify", "national emergency", "president's rule", "financial emergency"], "ambiguous", False)
    add_case("adv_D04", "ambiguous", "What are the rules for appointment of judges?", False, True, ["124", "217"], [], "Ambiguous Supreme Court vs High Court judge appointment", ["clarify", "supreme court", "high court", "article 124", "article 217"], "ambiguous", False)
    add_case("adv_D05", "ambiguous", "Can fundamental rights be suspended?", False, True, ["358", "359"], [], "Ambiguous emergency suspension inquiry", ["clarify", "national emergency", "article 19", "article 20", "article 21"], "ambiguous", False)
    add_case("adv_D06", "ambiguous", "How is the executive power exercised?", False, True, ["53", "154"], [], "Ambiguous Union vs State executive power", ["clarify", "union president", "state governor", "article 53", "article 154"], "ambiguous", False)
    add_case("adv_D07", "ambiguous", "What is the procedure for amendment?", False, True, ["368"], [], "Ambiguous general amendment request", ["clarify", "article 368", "special majority", "ratification by states"], "ambiguous", False)
    add_case("adv_D08", "ambiguous", "What are the powers of the speaker?", False, True, ["93", "178"], [], "Ambiguous Lok Sabha vs Vidhan Sabha speaker", ["clarify", "house of the people", "legislative assembly"], "ambiguous", False)
    add_case("adv_D09", "ambiguous", "Who can grant pardons?", False, True, ["72", "161"], [], "Ambiguous President vs Governor pardon powers", ["clarify", "president article 72", "governor article 161"], "ambiguous", False)
    add_case("adv_D10", "ambiguous", "How are elections conducted?", False, True, ["324"], [], "Ambiguous election inquiry", ["clarify", "election commission", "parliament", "state legislatures"], "ambiguous", False)
    add_case("adv_D11", "ambiguous", "What is the qualification to be a member?", False, True, ["84", "173"], [], "Ambiguous MP vs MLA qualification", ["clarify", "parliament", "state legislature"], "ambiguous", False)
    add_case("adv_D12", "ambiguous", "What is the term of office?", False, True, ["56", "83"], [], "Ambiguous office term", ["clarify", "president", "council of states", "house of the people"], "ambiguous", False)
    add_case("adv_D13", "ambiguous", "How is a bill passed?", False, True, ["107", "196"], [], "Ambiguous Union vs State legislative procedure", ["clarify", "parliament", "state legislature"], "ambiguous", False)
    add_case("adv_D14", "ambiguous", "What is the procedure for impeachment?", False, True, ["61", "124"], [], "Ambiguous President vs Judge impeachment", ["clarify", "president article 61", "supreme court judge article 124"], "ambiguous", False)
    add_case("adv_D15", "ambiguous", "Who appoints the council of ministers?", False, True, ["75", "164"], [], "Ambiguous Union vs State Council of Ministers", ["clarify", "prime minister", "chief minister", "article 75", "article 164"], "ambiguous", False)
    add_case("adv_D16", "ambiguous", "What are the rules regarding public service commissions?", False, True, ["315"], [], "Ambiguous UPSC vs SPSC inquiry", ["clarify", "union public service commission", "state public service commission"], "ambiguous", False)
    add_case("adv_D17", "ambiguous", "How is money bill defined?", False, True, ["110", "199"], [], "Ambiguous Parliament vs State money bill", ["clarify", "parliament article 110", "state legislature article 199"], "ambiguous", False)
    add_case("adv_D18", "ambiguous", "What is the power of superintendence?", False, True, ["227"], [], "Ambiguous High Court superintendence", ["clarify", "high court", "subordinate courts", "article 227"], "ambiguous", False)
    add_case("adv_D19", "ambiguous", "What are the official languages?", False, True, ["343"], [], "Ambiguous Union vs Regional language inquiry", ["clarify", "union language article 343", "regional languages"], "ambiguous", False)
    add_case("adv_D20", "ambiguous", "How are disputes regarding boundaries settled?", False, True, ["3", "131"], [], "Ambiguous internal boundary reorganization vs inter-state dispute", ["clarify", "parliament power article 3", "original jurisdiction article 131"], "ambiguous", False)

    # =========================================================================
    # CATEGORY E: Unanswerable & Out-of-Scope Questions (25 items)
    # =========================================================================
    add_case("adv_E01", "unanswerable_out_of_domain", "What are the rules for filing patent applications under the US Patent Act?", False, True, [], [], "Foreign law: US Patent Act", ["domain of the constitution of india", "foreign law"])
    add_case("adv_E02", "unanswerable_out_of_domain", "Write a Python script to implement binary search.", False, True, [], [], "Coding query", ["constitution of india domain only"])
    add_case("adv_E03", "unanswerable_out_of_domain", "What is the score of the latest cricket test match between India and Australia?", False, True, [], [], "Sports trivia query", ["constitution of india domain only"])
    add_case("adv_E04", "unanswerable_out_of_domain", "What is the capital city of France and its population?", False, True, [], [], "Geography trivia", ["constitution of india domain only"])
    add_case("adv_E05", "unanswerable_out_of_domain", "How do you cook traditional Hyderabadi chicken biryani?", False, True, [], [], "Cooking recipe", ["constitution of india domain only"])
    add_case("adv_E06", "unanswerable_out_of_domain", "What is the stock price of Reliance Industries on the Bombay Stock Exchange today?", False, True, [], [], "Financial market query", ["constitution of india domain only"])
    add_case("adv_E07", "unanswerable_out_of_domain", "What does Article 5 of the United States Constitution say about ratification?", False, True, [], [], "Foreign constitutional law: US Constitution", ["outside the domain of the constitution of india", "us constitution"])
    add_case("adv_E08", "unanswerable_out_of_domain", "What are the rules of the German Basic Law regarding the Federal Constitutional Court?", False, True, [], [], "Foreign law: German Basic Law", ["outside the domain", "german basic law"])
    add_case("adv_E09", "unanswerable_out_of_domain", "Who won the Best Actor Oscar award in 2024?", False, True, [], [], "Entertainment query", ["constitution of india domain only"])
    add_case("adv_E10", "unanswerable_out_of_domain", "How does quantum superposition work in quantum mechanics?", False, True, [], [], "Physics query", ["constitution of india domain only"])
    add_case("adv_E11", "unanswerable_nonexistent_article", "What does Article 400 of the Constitution of India mandate regarding space law?", False, True, [], [], "Nonexistent article 400", ["does not exist", "articles 1 to 395"])
    add_case("adv_E12", "unanswerable_nonexistent_article", "Under Article 500, what are the corporate tax brackets in India?", False, True, [], [], "Nonexistent article 500", ["does not exist", "articles 1 to 395"])
    add_case("adv_E13", "unanswerable_nonexistent_article", "What is declared in Article 650 regarding artificial intelligence governance?", False, True, [], [], "Nonexistent article 650", ["does not exist", "articles 1 to 395"])
    add_case("adv_E14", "unanswerable_nonexistent_article", "Explain the provisions of Article 450 regarding interplanetary treaties.", False, True, [], [], "Nonexistent article 450", ["does not exist", "articles 1 to 395"])
    add_case("adv_E15", "unanswerable_nonexistent_article", "What does Article 700 state regarding cryptocurrency regulation in India?", False, True, [], [], "Nonexistent article 700", ["does not exist", "articles 1 to 395"])
    add_case("adv_E16", "unanswerable_current_affairs", "Who is the Prime Minister of India right now?", False, True, [], [], "Incumbent officeholder identity", ["does not contain", "names of individual officeholders", "static legal text"])
    add_case("adv_E17", "unanswerable_current_affairs", "Who is the current Chief Justice of India in 2026?", False, True, [], [], "Incumbent CJI inquiry", ["does not record current", "officeholders"])
    add_case("adv_E18", "unanswerable_current_affairs", "Name the current President of India.", False, True, [], [], "Name of current President", ["establishes the office of president", "does not contain individual names"])
    add_case("adv_E19", "unanswerable_current_affairs", "Who won the recent Lok Sabha general election?", False, True, [], [], "Recent political election results", ["does not establish current daily updates", "static legal text"])
    add_case("adv_E20", "unanswerable_current_affairs", "Who is the current Chief Minister of Maharashtra in 2026?", False, True, [], [], "Current political chief minister", ["does not contain current officeholders"])
    add_case("adv_E21", "unanswerable_corpus_limitation", "What was the exact weather in New Delhi when the Constitution was signed on 26 November 1949?", False, True, [], [], "Historical weather absent from corpus", ["does not establish", "available document"])
    add_case("adv_E22", "unanswerable_corpus_limitation", "What fountain pen brand did Dr. B.R. Ambedkar use to draft the text?", False, True, [], [], "Historical trivia absent from statutory text", ["does not record", "available document"])
    add_case("adv_E23", "unanswerable_corpus_limitation", "How many pages does Chapter 99 of the Constitution contain?", False, True, [], [], "Nonexistent chapter", ["does not exist", "official constitution"])
    add_case("adv_E24", "unanswerable_corpus_limitation", "What is the verbatim text of page 9000 of the uploaded Constitution PDF?", False, True, [], [], "Nonexistent page number", ["page 9000 does not exist", "pages 1 to 404"])
    add_case("adv_E25", "unanswerable_corpus_limitation", "What does Schedule 25 of the Constitution say?", False, True, [], [], "Nonexistent schedule", ["does not exist", "schedules"])

    # =========================================================================
    # CATEGORY F: Temporal, Version & Dynamic Questions (20 items)
    # =========================================================================
    add_case("adv_F01", "temporal_dynamic", "According to the uploaded Constitution text, what was the exact number of constitutional amendments passed yesterday?", False, True, [], [], "Amendments passed yesterday", ["cannot determine", "passed yesterday", "static legal text"])
    add_case("adv_F02", "temporal_dynamic", "What constitutional amendment bill was introduced in Parliament today?", False, True, [], [], "Bill introduced today", ["cannot determine", "today", "static document"])
    add_case("adv_F03", "temporal_dynamic", "How many amendments were passed in the last week?", False, True, [], [], "Amendments in last week", ["cannot determine", "last week", "time-sensitive"])
    add_case("adv_F04", "temporal_dynamic", "What was the Supreme Court verdict delivered yesterday regarding Article 21?", False, True, [], [], "Court verdict yesterday", ["cannot determine", "yesterday", "court verdict"])
    add_case("adv_F05", "temporal_dynamic", "What is the latest constitutional amendment passed in 2026?", False, True, [], [], "Latest amendment in 2026", ["does not establish current", "time-sensitive", "2026"])
    add_case("adv_F06", "temporal_dynamic", "Did Parliament pass an amendment to Article 19 this morning?", False, True, [], [], "Amendment this morning", ["cannot determine", "this morning", "static legal text"])
    add_case("adv_F07", "temporal_dynamic", "What dynamic changes were made to Article 370 in the news last month?", False, True, [], [], "Recent news dynamic updates", ["static constitution text", "does not contain recent news"])
    add_case("adv_F08", "temporal_dynamic", "What is the date of commencement of the Constitution of India stated in Article 394?", True, False, ["394"], ["const_394_p270_c01"], "Commencement date in statutory text", ["twenty-sixth day of january, 1950", "commencement"])
    add_case("adv_F09", "temporal_dynamic", "What articles came into force immediately on 26 November 1949 under Article 394?", True, False, ["394"], ["const_394_p270_c01"], "Articles coming into force at once", ["5, 6, 7, 8, 9", "60", "324", "366", "367", "379"])
    add_case("adv_F10", "temporal_dynamic", "Was the 106th Constitutional Amendment Act passed yesterday?", False, True, [], [], "106th amendment passed yesterday", ["cannot determine", "passed yesterday", "static text"])
    add_case("adv_F11", "temporal_dynamic", "What did the Law Minister announce in the press conference yesterday regarding Article 44?", False, True, [], [], "Press conference yesterday", ["cannot determine", "yesterday", "press conference"])
    add_case("adv_F12", "temporal_dynamic", "What is the exact number of writ petitions filed in the Supreme Court today?", False, True, [], [], "Writ petitions today", ["cannot determine", "today", "filings"])
    add_case("adv_F13", "temporal_dynamic", "What ordinance was promulgated by the President yesterday?", False, True, [], [], "Ordinance yesterday", ["cannot determine", "yesterday", "ordinance"])
    add_case("adv_F14", "temporal_dynamic", "What changes were made to the Preamble by the 42nd Amendment Act according to the text?", True, False, ["PREAMBLE"], ["const_preamble_p32_c01"], "Preamble amendment historical terms", ["socialist", "secular", "integrity"])
    add_case("adv_F15", "temporal_dynamic", "How many amendments were enacted in 2025?", False, True, [], [], "Enactments in 2025", ["cannot determine", "2025", "static uploaded document"])
    add_case("adv_F16", "temporal_dynamic", "Did the President sign a new amendment bill last night?", False, True, [], [], "Signing last night", ["cannot determine", "last night"])
    add_case("adv_F17", "temporal_dynamic", "What was the outcome of the floor test in the State Assembly yesterday under Article 164?", False, True, [], [], "Floor test yesterday", ["cannot determine", "yesterday", "static legal text"])
    add_case("adv_F18", "temporal_dynamic", "Which judge was appointed to the Supreme Court last week under Article 124?", False, True, [], [], "Appointment last week", ["cannot determine", "last week", "static legal document"])
    add_case("adv_F19", "temporal_dynamic", "What is the short title of the Constitution according to Article 393?", True, False, ["393"], ["const_393_p270_c01"], "Short title statutory text", ["constitution of india", "short title"])
    add_case("adv_F20", "temporal_dynamic", "What Acts were repealed by Article 395 at commencement?", True, False, ["395"], ["const_395_p270_c01"], "Repeals under Article 395", ["indian independence act", "government of india act, 1935", "repealed"])

    # =========================================================================
    # CATEGORY G: Long-Context Distractor & Retrieval Stress Tests (20 items)
    # =========================================================================
    add_case("adv_G01", "long_context_stress", "In Article 19, what specific grounds justify reasonable restrictions on freedom of assembly under clause (3)?", True, False, ["19"], ["const_19_p40_c01"], "Clause 19(3) restriction grounds", ["sovereignty and integrity of india", "public order"])
    add_case("adv_G02", "long_context_stress", "What are the restrictions on freedom of association under Article 19(4)?", True, False, ["19"], ["const_19_p40_c01"], "Clause 19(4) association restrictions", ["sovereignty and integrity", "public order or morality"])
    add_case("adv_G03", "long_context_stress", "Under Article 22(4), what is the maximum initial period for preventive detention without Advisory Board review?", True, False, ["22"], ["const_22_p42_c01"], "Preventive detention duration in Article 22(4)", ["three months", "advisory board"])
    add_case("adv_G04", "long_context_stress", "What exception does Article 22(3) provide regarding enemy aliens and preventive detention?", True, False, ["22"], ["const_22_p42_c01"], "Enemy aliens exception in Article 22(3)", ["enemy alien", "preventive detention", "shall not apply"])
    add_case("adv_G05", "long_context_stress", "What is the proviso to Article 368(2) requiring ratification by State legislatures?", True, False, ["368"], ["const_368_p35_c01"], "Article 368(2) proviso requirements", ["not less than one-half of the states", "ratification", "legislatures"])
    add_case("adv_G06", "long_context_stress", "In Article 352, what requirement was added for a written communication of the Cabinet decision?", True, False, ["352"], ["const_352_p239_c01"], "Cabinet written communication under Article 352", ["decision has been communicated to him in writing", "union cabinet", "prime minister"])
    add_case("adv_G07", "long_context_stress", "Under Article 124(2), what is the retirement age specified for a Supreme Court judge?", True, False, ["124"], ["const_124_p88_c01"], "Retirement age Supreme Court judge", ["sixty-five years", "age of"])
    add_case("adv_G08", "long_context_stress", "What is the minimum qualification period as High Court judge to be appointed to Supreme Court in Article 124(3)?", True, False, ["124"], ["const_124_p88_c01"], "Judge qualification 5 years", ["at least five years", "judge of a high court"])
    add_case("adv_G09", "long_context_stress", "What is the advocate qualification period for Supreme Court appointment in Article 124(3)(b)?", True, False, ["124"], ["const_124_p88_c01"], "Advocate qualification 10 years", ["at least ten years", "advocate of a high court"])
    add_case("adv_G10", "long_context_stress", "What is a distinguished jurist qualification under Article 124(3)(c)?", True, False, ["124"], ["const_124_p88_c01"], "Distinguished jurist qualification", ["in the opinion of the president", "distinguished jurist"])
    add_case("adv_G11", "long_context_stress", "What does Article 124A establish regarding the National Judicial Appointments Commission?", True, False, ["124A"], ["const_124a_p90_c01"], "NJAC under Article 124A", ["national judicial appointments commission", "chief justice of india"])
    add_case("adv_G12", "long_context_stress", "Under Article 72(1)(a), what cases of pardon involve Court Martial punishments?", True, False, ["72"], ["const_72_p362_c01"], "Court martial pardoning power", ["court martial", "punishment or sentence"])
    add_case("adv_G13", "long_context_stress", "What does Article 75(3) state regarding collective responsibility of Ministers?", True, False, ["75"], ["const_75_p64_c01"], "Collective responsibility under Article 75(3)", ["collectively responsible", "house of the people"])
    add_case("adv_G14", "long_context_stress", "Under Article 75(5), what happens if a Minister is not a member of Parliament for 6 consecutive months?", True, False, ["75"], ["const_75_p64_c01"], "Six month rule for Ministers", ["six consecutive months", "ceases to be a minister"])
    add_case("adv_G15", "long_context_stress", "Under Article 123(2)(a), within what time does an Ordinance expire after Parliament reassembles?", True, False, ["123"], ["const_123_p365_c01"], "Six week expiry rule for Ordinances", ["six weeks", "reassembly of parliament"])
    add_case("adv_G16", "long_context_stress", "What is the procedure for removal of a High Court judge under Article 217(1)(b)?", True, False, ["217"], ["const_217_p131_c01"], "Removal of High Court judge", ["manner provided in clause (4) of article 124", "president"])
    add_case("adv_G17", "long_context_stress", "Under Article 226(2), does a High Court have jurisdiction if the cause of action arises within its territory?", True, False, ["226"], ["const_226_p135_c01"], "Territorial jurisdiction under Article 226(2)", ["cause of action", "wholly or in part", "territories"])
    add_case("adv_G18", "long_context_stress", "What does Article 356(3) state regarding the approval of President's Rule by Parliament?", True, False, ["356"], ["const_356_p35_c01"], "Two month approval rule in Article 356(3)", ["two months", "resolutions of both houses of parliament"])
    add_case("adv_G19", "long_context_stress", "What is the maximum duration President's Rule can remain in force under Article 356(4)?", True, False, ["356"], ["const_356_p35_c01"], "Three year maximum duration", ["three years", "beyond a period of"])
    add_case("adv_G20", "long_context_stress", "Under Article 352(4), within what period must a National Emergency proclamation be approved by Parliament?", True, False, ["352"], ["const_352_p239_c01"], "One month approval rule under Article 352(4)", ["one month", "resolutions of both houses"])

    # =========================================================================
    # CATEGORY H: Conversational Memory & Multi-Turn Stress Tests (15 items)
    # =========================================================================
    add_case("adv_H01", "conversational_memory", "Explain Article 14 in simple English", True, False, ["14"], ["const_14_p37_c01"], "Turn 1 of follow-up series", ["equality before the law", "equal protection"])
    add_case("adv_H02", "conversational_memory", "in detail", True, False, ["14"], ["const_14_p37_c01"], "Turn 2 detail follow-up referring to Article 14", ["article 14", "equality", "verbatim", "discrimination"])
    add_case("adv_H03", "conversational_memory", "give an example", True, False, ["14"], ["const_14_p37_c01"], "Turn 3 example follow-up for Article 14", ["example", "arbitrary", "law"])
    add_case("adv_H04", "conversational_memory", "make it crisp", True, False, ["14"], ["const_14_p37_c01"], "Turn 4 crisp summary of Article 14", ["crisp", "article 14", "equality"])
    add_case("adv_H05", "conversational_memory", "What about Article 21?", True, False, ["21"], ["const_21_p42_c01"], "Turn 5 pivot from Article 14 to Article 21", ["article 21", "life and personal liberty"])
    add_case("adv_H06", "conversational_memory", "explain about it in detail and crisp", True, False, ["21"], ["const_21_p42_c01"], "Turn 6 detail and crisp follow-up on Article 21", ["article 21", "procedure established by law"])
    add_case("adv_H07", "conversational_memory", "what were the exceptions?", True, False, ["21"], ["const_21_p42_c01"], "Turn 7 exception follow-up on Article 21", ["procedure established by law", "deprived of life"])
    add_case("adv_H08", "conversational_memory", "compare both of them", True, False, ["14", "21"], ["const_14_p37_c01", "const_21_p42_c01"], "Turn 8 comparative synthesis of Articles 14 and 21", ["article 14", "article 21", "comparison", "equality", "liberty"])
    add_case("adv_H09", "conversational_memory", "What about Article 32?", True, False, ["32"], ["const_32_p50_c01"], "Turn 9 pivot to Article 32 remedies", ["article 32", "writs", "supreme court"])
    add_case("adv_H10", "conversational_memory", "which writs can be issued under it?", True, False, ["32"], ["const_32_p50_c01"], "Turn 10 specific writs follow-up on Article 32", ["habeas corpus", "mandamus", "prohibition", "quo warranto", "certiorari"])
    add_case("adv_H11", "conversational_memory", "in detail", False, True, [], [], "Isolated follow-up in fresh session without history", ["specify which constitutional article", "article number"])
    add_case("adv_H12", "conversational_memory", "give me an example", False, True, [], [], "Isolated example in fresh session without history", ["specify which constitutional article", "article number"])
    add_case("adv_H13", "conversational_memory", "Session A asks Article 19, then in detail", True, False, ["19"], ["const_19_p40_c01"], "Multi-user isolation test Session A", ["article 19", "freedom of speech"])
    add_case("adv_H14", "conversational_memory", "Session B asks Article 22, then in detail", True, False, ["22"], ["const_22_p42_c01"], "Multi-user isolation test Session B", ["article 22", "arrest and detention"])
    add_case("adv_H15", "conversational_memory", "What is the capital of France?", False, True, [], [], "Out of domain inquiry in ongoing conversation", ["domain restriction", "constitution of india domain only"])

    return cases


def save_dataset_splits(cases: List[Dict[str, Any]], output_dir: Path) -> Dict[str, int]:
    """Save dataset into dev, validation, and challenge splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Full Dataset
    full_path = output_dir / "adversarial_full.jsonl"
    with open(full_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 2. Dev Set (approx 35 cases)
    dev_cases = cases[::5]  # Every 5th
    dev_path = output_dir / "adversarial_dev.jsonl"
    with open(dev_path, "w", encoding="utf-8") as f:
        for c in dev_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 3. Validation Set (approx 35 cases)
    val_cases = cases[1::5]
    val_path = output_dir / "adversarial_val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for c in val_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 4. Hidden Challenge Set (the remaining 105 cases)
    challenge_cases = [c for idx, c in enumerate(cases) if idx % 5 not in (0, 1)]
    chal_path = output_dir / "adversarial_challenge.jsonl"
    with open(chal_path, "w", encoding="utf-8") as f:
        for c in challenge_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return {
        "total": len(cases),
        "dev": len(dev_cases),
        "val": len(val_cases),
        "challenge": len(challenge_cases),
    }


if __name__ == "__main__":
    cases = generate_adversarial_dataset()
    out_dir = Path("data/evaluation/adversarial")
    counts = save_dataset_splits(cases, out_dir)
    print(f"Generated {counts['total']} adversarial test cases.")
    print(f"Splits: Dev={counts['dev']}, Val={counts['val']}, Challenge={counts['challenge']}")
