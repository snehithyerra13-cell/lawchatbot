"""Article parser for Constitution of India.

Parses extracted PDF pages into structured constitutional records (Articles,
Preamble, Schedules, Parts) with titles, text, page numbers, plain-English
explanations, practical examples, and searchable keywords.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Curated high-value constitutional knowledge repository
# Grounded in Indian constitutional jurisprudence
LANDMARK_PROVISIONS: Dict[str, Dict[str, Any]] = {
    "PREAMBLE": {
        "title": "Preamble to the Constitution of India",
        "part": "Preamble",
        "part_title": "Preamble",
        "simple_explanation": (
            "The Preamble is the guiding spirit and introduction to the Constitution of India. "
            "It declares India to be a Sovereign, Socialist, Secular, Democratic Republic and commits "
            "to securing for all citizens Justice (social, economic, and political), Liberty (of thought, "
            "expression, belief, faith, and worship), Equality (of status and opportunity), and "
            "Fraternity (assuring dignity and unity of the Nation). As held in Kesavananda Bharati (1973), "
            "the Preamble is an integral part of the Constitution and reflects its Basic Structure."
        ),
        "examples": [
            "A citizen facing economic inequality or discrimination can invoke constitutional values rooted in the Preamble's commitment to social justice.",
            "Judges use the Preamble to interpret ambiguous provisions of statutes in harmony with democratic and secular ideals."
        ],
        "key_principles": [
            "Sovereign Socialist Secular Democratic Republic",
            "Justice, Liberty, Equality, and Fraternity",
            "Basic Structure Doctrine",
            "Guiding light for constitutional interpretation"
        ],
        "keywords": ["preamble", "sovereign", "socialist", "secular", "democratic", "republic", "justice", "liberty", "equality", "fraternity"]
    },
    "1": {
        "title": "Name and territory of the Union",
        "part": "PART I",
        "part_title": "THE UNION AND ITS TERRITORY",
        "simple_explanation": (
            "Article 1 establishes that 'India, that is Bharat, shall be a Union of States.' "
            "It signifies that the Indian federation is an indestructible union of destructible states, "
            "meaning states have no right to secede from the Union of India."
        ),
        "examples": [
            "States like Maharashtra, Tamil Nadu, or Punjab cannot legally declare independence or secede from India.",
            "Parliament can reorganize internal state borders under Article 3 without breaking the Union."
        ],
        "key_principles": [
            "Indestructible Union of States",
            "Territorial integrity of India",
            "Dual names: India and Bharat"
        ],
        "keywords": ["india", "bharat", "union of states", "territory", "federation"]
    },
    "14": {
        "title": "Equality before law",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 14 guarantees two vital rights to all persons within India: "
            "(1) Equality before the law (no person is above the law), and "
            "(2) Equal protection of the laws (equals must be treated equally). "
            "It strictly forbids arbitrariness in state action and requires that any classification "
            "made by the government must be based on an intelligible differentia with a rational nexus "
            "to the objective sought."
        ),
        "examples": [
            "The government cannot arbitrarily grant public contracts or licenses to favored individuals without an open, fair, and transparent process (E.P. Royappa doctrine).",
            "The state can create special legal protections for women, children, or socially disadvantaged groups because treating unequals equally is itself a form of inequality."
        ],
        "key_principles": [
            "Rule of Law",
            "Non-arbitrariness (E.P. Royappa v. State of Tamil Nadu)",
            "Doctrine of Reasonable Classification (Intelligible Differentia & Rational Nexus)",
            "Applies to all persons, both citizens and non-citizens"
        ],
        "keywords": ["equality", "rule of law", "arbitrary", "discrimination", "equal protection", "classification", "article 14"]
    },
    "15": {
        "title": "Prohibition of discrimination on grounds of religion, race, caste, sex or place of birth",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 15 prohibits the State from discriminating against any citizen solely on grounds of religion, "
            "race, caste, sex, or place of birth. It ensures equal access to public shops, restaurants, wells, and places "
            "of public resort. Importantly, clauses (3), (4), (5), and (6) allow affirmative action (special provisions and "
            "reservations) for women, children, SCs, STs, OBCs, and economically weaker sections (EWS)."
        ),
        "examples": [
            "A public restaurant cannot deny entry to a person on the basis of caste or religion.",
            "The government can reserve seats in educational institutions for women or socially/economically disadvantaged classes."
        ],
        "key_principles": [
            "Non-discrimination on specified grounds",
            "Public access rights",
            "Substantive equality and affirmative action"
        ],
        "keywords": ["discrimination", "religion", "race", "caste", "sex", "place of birth", "reservations", "affirmative action"]
    },
    "17": {
        "title": "Abolition of Untouchability",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 17 completely abolishes 'Untouchability' and forbids its practice in any form. "
            "It is an absolute fundamental right enforceable against private individuals as well as the State. "
            "Enforcement of any disability arising out of untouchability is a punishable criminal offence under the "
            "Protection of Civil Rights Act, 1955 and the SC/ST (Prevention of Atrocities) Act, 1989."
        ),
        "examples": [
            "Denying a person entry to a public temple, village water well, or shop on grounds of caste is a direct violation of Article 17.",
            "Refusing to serve food or sit with someone in a public restaurant based on untouchability is legally prohibited and punishable."
        ],
        "key_principles": [
            "Absolute Abolition of Untouchability",
            "Horizontal application (enforceable against private citizens)",
            "Enforced via Protection of Civil Rights Act, 1955",
            "Non-derogable social equality right"
        ],
        "keywords": ["untouchability", "abolition of untouchability", "caste discrimination", "article 17", "dalit rights", "protection against untouchability"]
    },
    "19": {
        "title": "Protection of certain rights regarding freedom of speech, etc.",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 19(1) guarantees six democratic freedoms to Indian citizens: "
            "(a) Freedom of speech and expression, (b) Peaceful assembly without arms, "
            "(c) Forming associations or unions, (d) Moving freely throughout India, "
            "(e) Residing and settling in any part of India, and (g) Practising any profession, trade, or business. "
            "These rights are not absolute; under clauses (2) to (6), the State can impose reasonable restrictions "
            "on grounds such as national security, public order, decency, morality, or defamation."
        ),
        "examples": [
            "Citizens and journalists have the right to express political opinions and criticize government policies peacefully.",
            "A peaceful candlelight march or protest in a public park is protected under freedom of peaceful assembly, provided it does not incite violence or disrupt public order."
        ],
        "key_principles": [
            "Six Democratic Freedoms",
            "Reasonable Restrictions Doctrine",
            "Proportionality test (Modern jurisprudence)",
            "Rights exclusive to citizens of India"
        ],
        "keywords": ["freedom of speech", "expression", "peaceful assembly", "association", "movement", "profession", "reasonable restrictions"]
    },
    "20": {
        "title": "Protection in respect of conviction for offences",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 20 provides three fundamental criminal law safeguards: "
            "(1) Protection against ex-post facto laws (no person can be punished for an act that was not an offence when committed), "
            "(2) Protection against double jeopardy (no person can be prosecuted and punished twice for the same offence), and "
            "(3) Protection against self-incrimination (no accused can be compelled to be a witness against themselves)."
        ),
        "examples": [
            "If an act is made a crime by a new statute today, a citizen who committed that act yesterday cannot be prosecuted under the new statute.",
            "Police cannot force an accused suspect to confess to a crime under torture or threat."
        ],
        "key_principles": [
            "No Ex-Post Facto Law",
            "Double Jeopardy (Nemo debet bis vexari)",
            "Right against Self-Incrimination",
            "Cannot be suspended even during Emergency (Article 359)"
        ],
        "keywords": ["double jeopardy", "self-incrimination", "ex post facto", "conviction", "offences", "criminal safeguard"]
    },
    "21": {
        "title": "Protection of life and personal liberty",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 21 provides that 'No person shall be deprived of his life or personal liberty except "
            "according to procedure established by law.' Through landmark rulings (especially Maneka Gandhi v. "
            "Union of India, 1978), the Supreme Court held that the procedure must be 'just, fair, and reasonable' "
            "(incorporating procedural due process). The Court has expanded 'life' far beyond mere animal existence "
            "to include human dignity, privacy, clean environment, health, shelter, and legal aid."
        ),
        "examples": [
            "Right to Privacy: Citizens are protected from unwarranted state surveillance and unauthorized tracking of personal communications (K.S. Puttaswamy judgment, 2017).",
            "Right to Healthcare & Environment: Citizens can petition the courts if industrial pollution contaminates public drinking water, violating their right to a healthy life.",
            "Protection against Arbitrary Arrest: Police cannot detain someone without lawful justification and humane treatment."
        ],
        "key_principles": [
            "Life with Human Dignity (Francis Coralie Mullin case)",
            "Just, Fair, and Reasonable Procedure (Maneka Gandhi v. Union of India)",
            "Right to Privacy (K.S. Puttaswamy v. Union of India)",
            "Non-derogable right: Cannot be suspended even during National Emergency (Article 359)",
            "Applies to all persons, citizens and foreigners alike"
        ],
        "keywords": ["life", "personal liberty", "privacy", "dignity", "procedure established by law", "maneka gandhi", "puttaswamy", "article 21"]
    },
    "21A": {
        "title": "Right to education",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 21A mandates that the State shall provide free and compulsory education to all children "
            "aged 6 to 14 years in such manner as the State may determine by law. Introduced by the 86th Constitutional "
            "Amendment Act (2002), it led to the enactment of the Right of Children to Free and Compulsory Education (RTE) Act, 2009."
        ),
        "examples": [
            "A government school cannot deny admission to an 8-year-old child or demand tuition fees.",
            "Private unaided schools are required under the RTE Act to reserve 25% of entry-level seats for children from disadvantaged and economically weaker sections."
        ],
        "key_principles": [
            "Fundamental Right to Elementary Education",
            "Ages 6 to 14 Years",
            "86th Constitutional Amendment Act, 2002",
            "Enforced via Right to Education (RTE) Act"
        ],
        "keywords": ["right to education", "rte", "free education", "compulsory education", "children", "86th amendment", "article 21a"]
    },
    "22": {
        "title": "Protection against arrest and detention in certain cases",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 22 guarantees procedural rights to any arrested person: "
            "(1) Right to be informed of the grounds of arrest as soon as possible, "
            "(2) Right to consult and be defended by a legal practitioner of choice, and "
            "(3) Right to be produced before the nearest magistrate within 24 hours of arrest. "
            "It also regulates preventive detention laws, imposing strict safeguards including Advisory Boards."
        ),
        "examples": [
            "If police arrest a person, they must produce them before a judicial magistrate within 24 hours (excluding journey time) or release them.",
            "An arrested suspect has the right to contact their lawyer before interrogation."
        ],
        "key_principles": [
            "24-Hour Magistrate Production Rule",
            "Right to Legal Counsel",
            "Safeguards against Preventive Detention"
        ],
        "keywords": ["arrest", "detention", "magistrate", "24 hours", "lawyer", "preventive detention"]
    },
    "25": {
        "title": "Freedom of conscience and free profession, practice and propagation of religion",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 25 guarantees to all persons freedom of conscience and the right freely to profess, practise, "
            "and propagate religion, subject to public order, morality, health, and other Fundamental Rights. "
            "It also permits the State to regulate economic, financial, or secular activities associated with religious practice "
            "and provide for social welfare and reform of Hindu religious institutions."
        ),
        "examples": [
            "An individual has the right to follow their faith, pray, and wear religious symbols (e.g. wearing of kirpan by Sikhs) peacefully.",
            "State laws banning untouchability or reforming temple entry for all castes are protected under Article 25(2)(b)."
        ],
        "key_principles": [
            "Freedom of Conscience",
            "Profess, Practise, and Propagate",
            "Subject to Public Order, Morality, and Health",
            "Essential Religious Practices Doctrine"
        ],
        "keywords": ["religion", "freedom of conscience", "practice", "propagate", "secularism", "faith"]
    },
    "32": {
        "title": "Remedies for enforcement of rights conferred by this Part",
        "part": "PART III",
        "part_title": "FUNDAMENTAL RIGHTS",
        "simple_explanation": (
            "Article 32 gives citizens the right to petition the Supreme Court of India directly for the enforcement "
            "of Fundamental Rights. Dr. B.R. Ambedkar called Article 32 the 'very soul of the Constitution and the very "
            "heart of it.' Under Article 32, the Supreme Court can issue prerogative writs: Habeas Corpus (release illegal detainee), "
            "Mandamus (command public duty), Prohibition (halt lower court excess), Quo Warranto (challenge illegal public office), "
            "and Certiorari (quash illegal administrative/judicial orders)."
        ),
        "examples": [
            "If a citizen is unlawfully held in jail without charges, their family can file a Habeas Corpus writ directly in the Supreme Court under Article 32 for immediate release.",
            "Public Interest Litigation (PIL) was born under Article 32, allowing activists to seek judicial intervention on behalf of bonded laborers, prisoners, or polluted rivers."
        ],
        "key_principles": [
            "Heart and Soul of the Constitution (Dr. B.R. Ambedkar)",
            "Prerogative Writs (Habeas Corpus, Mandamus, Prohibition, Quo Warranto, Certiorari)",
            "Direct Access to Supreme Court",
            "Basic Structure of the Constitution"
        ],
        "keywords": ["article 32", "constitutional remedies", "writs", "habeas corpus", "mandamus", "supreme court", "heart and soul", "pil"]
    },
    "44": {
        "title": "Uniform civil code for the citizens",
        "part": "PART IV",
        "part_title": "DIRECTIVE PRINCIPLES OF STATE POLICY",
        "simple_explanation": (
            "Article 44 states that 'The State shall endeavour to secure for the citizens a uniform civil code "
            "throughout the territory of India.' As a Directive Principle of State Policy (DPSP), it is a constitutional "
            "goal aimed at unifying personal laws governing marriage, divorce, inheritance, and adoption across communities, "
            "though DPSPs are non-justiciable (cannot be directly enforced by court mandamus)."
        ),
        "examples": [
            "State or national legislatures drafting common family laws regarding equal inheritance rights for daughters across all religious communities invoke Article 44."
        ],
        "key_principles": [
            "Directive Principle of State Policy",
            "Uniform civil code (UCC)",
            "Non-justiciable directive to the State"
        ],
        "keywords": ["uniform civil code", "ucc", "article 44", "personal law", "directive principles", "dpsp"]
    },
    "51A": {
        "title": "Fundamental duties",
        "part": "PART IVA",
        "part_title": "FUNDAMENTAL DUTIES",
        "simple_explanation": (
            "Article 51A prescribes 11 Fundamental Duties for every citizen of India. Inserted by the 42nd Amendment (1976) "
            "on the recommendation of the Swaran Singh Committee (with an 11th duty added by the 86th Amendment in 2002), "
            "they urge citizens to respect the National Flag and Anthem, cherish the freedom struggle, defend the country, "
            "promote harmony and brotherhood, renounce practices derogatory to women, protect the environment, develop scientific temper, "
            "and provide educational opportunities to children aged 6 to 14."
        ),
        "examples": [
            "Standing respectfully during the National Anthem and protecting public property during rallies reflects Article 51A duties.",
            "Planting trees, protecting lakes and wildlife, and avoiding littering correspond to duty 51A(g) to protect natural environment."
        ],
        "key_principles": [
            "Part IVA / Swaran Singh Committee Recommendations",
            "42nd and 86th Constitutional Amendments",
            "Moral and civic obligations of Indian citizens",
            "Non-enforceable by direct writ, but used by courts to interpret statutory reasonableness"
        ],
        "keywords": ["fundamental duties", "duty", "51a", "national flag", "environment", "scientific temper", "swaran singh", "citizenship"]
    },
    "52": {
        "title": "The President of India",
        "part": "PART V",
        "part_title": "THE UNION",
        "simple_explanation": (
            "Article 52 concisely establishes the highest constitutional office in the republic: "
            "'There shall be a President of India.' The President is the constitutional Head of State and the first citizen of India."
        ),
        "examples": [
            "All executive actions of the Government of India are formally taken in the name of the President of India (under Article 77)."
        ],
        "key_principles": [
            "Head of State",
            "Constitutional Monarch / Republic President equivalent",
            "Executive power vested in President under Article 53"
        ],
        "keywords": ["president", "president of india", "article 52", "head of state", "executive"]
    },
    "74": {
        "title": "Council of Ministers to aid and advise President",
        "part": "PART V",
        "part_title": "THE UNION",
        "simple_explanation": (
            "Article 74 establishes the parliamentary system of governance in India. It provides that there shall be a "
            "Council of Ministers with the Prime Minister at the head to aid and advise the President, who shall, in the "
            "exercise of his functions, act in accordance with such advice. The President may require the Council of Ministers "
            "to reconsider such advice once, but must act according to the reconsidered advice."
        ),
        "examples": [
            "When bills are passed by Parliament, the President gives assent on the binding advice of the Union Cabinet headed by the Prime Minister."
        ],
        "key_principles": [
            "Parliamentary Democracy",
            "De jure Head (President) vs De facto Head (Prime Minister)",
            "Binding nature of Cabinet advice (42nd and 44th Amendments)"
        ],
        "keywords": ["council of ministers", "prime minister", "aid and advise", "president", "cabinet", "article 74"]
    },
    "75": {
        "title": "Other provisions as to Ministers",
        "part": "PART V",
        "part_title": "THE UNION",
        "simple_explanation": (
            "Article 75 governs the appointment and tenure of Union Ministers: "
            "(1) The Prime Minister is appointed by the President, and other Ministers are appointed on the PM's advice, "
            "(2) Ministers hold office during the pleasure of the President, "
            "(3) The Council of Ministers is collectively responsible to the House of the People (Lok Sabha), and "
            "(4) Total number of Ministers cannot exceed 15% of Lok Sabha strength (91st Amendment)."
        ),
        "examples": [
            "If the Lok Sabha passes a Vote of No Confidence against the government, the entire Council of Ministers including the Prime Minister must resign."
        ],
        "key_principles": [
            "Collective Responsibility to Lok Sabha",
            "Appointment of Prime Minister and Ministers",
            "15% Ministerial Cap (91st Amendment)"
        ],
        "keywords": ["prime minister", "ministers", "collective responsibility", "lok sabha", "no confidence", "article 75"]
    },
    "124": {
        "title": "Establishment and constitution of Supreme Court",
        "part": "PART V",
        "part_title": "THE UNION",
        "simple_explanation": (
            "Article 124 establishes the Supreme Court of India consisting of the Chief Justice of India (CJI) and other judges. "
            "It outlines judge qualifications (must be an Indian citizen and have served as a High Court judge for 5 years or advocate "
            "for 10 years or be a distinguished jurist), retirement age (65 years), and stringent impeachment removal procedures "
            "requiring a two-thirds majority in both Houses of Parliament on grounds of proved misbehaviour or incapacity."
        ),
        "examples": [
            "A Supreme Court judge cannot be removed arbitrarily by the executive; removal requires a rigorous impeachment motion in Parliament under Article 124(4)."
        ],
        "key_principles": [
            "Independence of Judiciary",
            "Appointment and Collegium System",
            "Security of Judicial Tenure",
            "Removal only by Parliamentary Impeachment"
        ],
        "keywords": ["supreme court", "chief justice", "judges", "collegium", "impeachment", "article 124", "judiciary"]
    },
    "226": {
        "title": "Power of High Courts to issue certain writs",
        "part": "PART VI",
        "part_title": "THE STATES",
        "simple_explanation": (
            "Article 226 empowers High Courts to issue prerogative writs (Habeas Corpus, Mandamus, Prohibition, Quo Warranto, "
            "Certiorari) for the enforcement of Fundamental Rights and 'for any other purpose' (such as ordinary legal and statutory rights). "
            "Because of the phrase 'for any other purpose', the writ jurisdiction of High Courts under Article 226 is technically wider "
            "in subject matter than that of the Supreme Court under Article 32."
        ),
        "examples": [
            "A citizen can approach the state High Court if a municipal corporation unlawfully demolishes their shop without statutory notice, even if no Fundamental Right is violated directly."
        ],
        "key_principles": [
            "High Court Writ Jurisdiction",
            "Wider scope than Article 32 ('for any other purpose')",
            "Discretionary constitutional remedy"
        ],
        "keywords": ["high court", "article 226", "writs", "habeas corpus", "mandamus", "any other purpose"]
    },
    "352": {
        "title": "Proclamation of Emergency (National Emergency)",
        "part": "PART XVIII",
        "part_title": "EMERGENCY PROVISIONS",
        "simple_explanation": (
            "Article 352 allows the President to proclaim a National Emergency if the security of India or any part of its territory "
            "is threatened by war, external aggression, or 'armed rebellion' (changed from 'internal disturbance' by the 44th Amendment, 1978). "
            "The proclamation can only be issued upon the written recommendation of the Union Cabinet and must be approved by Parliament "
            "within one month by a special majority."
        ),
        "examples": [
            "During a national war or severe external invasion, a proclamation under Article 352 transforms India's federal structure into a unitary system, allowing the Centre to direct state executive actions."
        ],
        "key_principles": [
            "National Emergency grounds: War, External Aggression, Armed Rebellion",
            "Mandatory Written Cabinet Advice (44th Amendment)",
            "Special Parliamentary Approval within 1 Month",
            "Impact on Fundamental Rights (Articles 20 and 21 cannot be suspended)"
        ],
        "keywords": ["national emergency", "article 352", "war", "external aggression", "armed rebellion", "emergency"]
    },
    "356": {
        "title": "Provisions in case of failure of constitutional machinery in States (President's Rule)",
        "part": "PART XVIII",
        "part_title": "EMERGENCY PROVISIONS",
        "simple_explanation": (
            "Article 356 allows the President to impose President's Rule in a State if satisfied, on receiving a report from "
            "the Governor or otherwise, that the governance of the State cannot be carried on in accordance with the Constitution. "
            "In the landmark S.R. Bommai case (1994), the Supreme Court ruled that Article 356 is subject to judicial review, and the "
            "state assembly cannot be dissolved until Parliament approves the proclamation."
        ),
        "examples": [
            "If no political party or coalition can prove a legislative majority on the floor of the state assembly after an election, President's Rule may be imposed."
        ],
        "key_principles": [
            "President's Rule / State Emergency",
            "Subject to Judicial Review (S.R. Bommai v. Union of India)",
            "Floor Test is the only litmus test of majority",
            "Maximum duration normally up to 3 years with periodic approvals"
        ],
        "keywords": ["presidents rule", "state emergency", "article 356", "constitutional machinery", "governor", "bommai"]
    },
    "368": {
        "title": "Power of Parliament to amend the Constitution and procedure therefor",
        "part": "PART XX",
        "part_title": "AMENDMENT OF THE CONSTITUTION",
        "simple_explanation": (
            "Article 368 details how the Constitution of India can be amended. An amendment bill can be initiated in either "
            "House of Parliament and must be passed by a special majority (majority of total membership and at least two-thirds of members "
            "present and voting). Amendments affecting federal provisions (e.g., representation of states, Supreme Court powers) also require "
            "ratification by at least half of State Legislatures. Crucially, under the Kesavananda Bharati doctrine (1973), Parliament "
            "cannot amend or destroy the 'Basic Structure' of the Constitution."
        ),
        "examples": [
            "The 101st Constitutional Amendment introducing the Goods and Services Tax (GST) required a special majority in Parliament plus ratification by more than half the state assemblies.",
            "If Parliament passes an amendment abolishing judicial review or judicial independence, the Supreme Court can strike it down as unconstitutional under the Basic Structure doctrine (as in the NJAC judgment)."
        ],
        "key_principles": [
            "Constituent Power of Parliament",
            "Special Majority & State Ratification",
            "Basic Structure Limitation (Kesavananda Bharati v. State of Kerala)",
            "No joint sitting permitted for constitutional amendment bills"
        ],
        "keywords": ["amendment", "article 368", "basic structure", "special majority", "ratification", "constitution amendment"]
    },
    "370": {
        "title": "Temporary provisions with respect to the State of Jammu and Kashmir",
        "part": "PART XXI",
        "part_title": "TEMPORARY, TRANSITIONAL AND SPECIAL PROVISIONS",
        "simple_explanation": (
            "Article 370 was a temporary provision granting special autonomous status to the State of Jammu and Kashmir, "
            "limiting Parliament's power to make laws for the state primarily to defence, foreign affairs, and communications unless concurred "
            "by the state government. On 5th and 6th August 2019, through Presidential Orders (C.O. 272 and C.O. 273) and the Jammu and Kashmir "
            "Reorganisation Act, 2019, all clauses of Article 370 were rendered inoperative, and the Constitution of India became fully applicable "
            "to Jammu and Kashmir without exception. The Supreme Court upheld the abrogation in December 2023."
        ),
        "examples": [
            "Following the 2019 presidential declarations (documented in Appendix III of the loaded Constitution PDF), all Indian central laws and Fundamental Rights apply equally to Jammu, Kashmir, and Ladakh."
        ],
        "key_principles": [
            "Originally framed as 'Temporary provision'",
            "Rendered inoperative via C.O. 272 & C.O. 273 (August 2019)",
            "Upheld by Supreme Court Constitution Bench (December 2023)"
        ],
        "keywords": ["article 370", "jammu and kashmir", "special status", "temporary provision", "abrogation", "c.o. 273"]
    }
}

# Thematic Topic Directory mapping constitutional topics to articles and parts
TOPIC_CATALOG: Dict[str, Dict[str, Any]] = {
    "fundamental_rights": {
        "title": "Fundamental Rights (Part III, Articles 12-35)",
        "part": "PART III",
        "article_range": "12 to 35",
        "key_articles": ["14", "15", "16", "17", "18", "19", "20", "21", "21A", "22", "23", "24", "25", "29", "30", "32"],
        "overview": (
            "Part III of the Constitution of India (Articles 12 to 35) embodies the Fundamental Rights, "
            "often described as the Magna Carta of India. These rights guarantee civil liberties and equal protection "
            "against arbitrary state action. They are classified into six broad categories: (1) Right to Equality (Articles 14-18), "
            "(2) Right to Freedom (Articles 19-22), (3) Right against Exploitation (Articles 23-24), "
            "(4) Right to Freedom of Religion (Articles 25-28), (5) Cultural and Educational Rights (Articles 29-30), and "
            "(6) Right to Constitutional Remedies (Articles 32-35). Fundamental rights are directly enforceable against the State."
        ),
        "keywords": ["fundamental rights", "rights", "part iii", "equality", "freedom", "liberty", "remedies", "writs"]
    },
    "directive_principles": {
        "title": "Directive Principles of State Policy (Part IV, Articles 36-51)",
        "part": "PART IV",
        "article_range": "36 to 51",
        "key_articles": ["38", "39", "39A", "40", "44", "45", "48A", "50", "51"],
        "overview": (
            "Part IV of the Constitution (Articles 36 to 51) sets out the Directive Principles of State Policy (DPSP), "
            "inspired by the Irish Constitution. They represent socio-economic goals intended to establish a welfare state "
            "and economic democracy. While Article 37 states that DPSPs are non-justiciable (courts cannot force the government to "
            "enact them), they are fundamental in the governance of the country, and the State is constitutionally obligated to apply "
            "these principles in making laws."
        ),
        "keywords": ["directive principles", "dpsp", "part iv", "welfare state", "uniform civil code", "article 44", "panchayats"]
    },
    "fundamental_duties": {
        "title": "Fundamental Duties (Part IVA, Article 51A)",
        "part": "PART IVA",
        "article_range": "51A",
        "key_articles": ["51A"],
        "overview": (
            "Part IVA was added to the Constitution by the 42nd Amendment Act (1976) upon the recommendation of the Swaran Singh "
            "Committee. It contains Article 51A, which enumerates 11 fundamental duties expected of every citizen, including respecting "
            "the National Flag and National Anthem, safeguarding public property, protecting the environment, renouncing practices "
            "derogatory to women, and promoting scientific temper."
        ),
        "keywords": ["fundamental duties", "duties", "51a", "part iva", "citizenship duties", "swaran singh"]
    },
    "president_and_executive": {
        "title": "The Union Executive — President, Vice-President & Council of Ministers",
        "part": "PART V",
        "article_range": "52 to 78",
        "key_articles": ["52", "53", "54", "61", "72", "74", "75", "76"],
        "overview": (
            "Part V (Chapter I) establishes the Union Executive. The President of India (Article 52) is the ceremonial Head of State, "
            "in whose name all executive actions are taken. However, India is a parliamentary democracy where the President acts on the "
            "aid and advice of the Council of Ministers headed by the Prime Minister (Article 74). Article 75 establishes collective "
            "responsibility of the Council of Ministers to the Lok Sabha."
        ),
        "keywords": ["president", "prime minister", "executive", "council of ministers", "cabinet", "part v"]
    },
    "parliament": {
        "title": "Parliament of India (Part V, Articles 79-122)",
        "part": "PART V",
        "article_range": "79 to 122",
        "key_articles": ["79", "80", "81", "100", "105", "107", "110", "112"],
        "overview": (
            "Part V (Chapter II) provides for the Union Parliament, consisting of the President and two Houses: "
            "the Council of States (Rajya Sabha) and the House of the People (Lok Sabha). It governs qualifications, "
            "legislative procedure, Money Bills (Article 110), Annual Financial Statement / Budget (Article 112), and "
            "parliamentary privileges (Article 105)."
        ),
        "keywords": ["parliament", "lok sabha", "rajya sabha", "money bill", "budget", "legislation", "speaker"]
    },
    "judiciary": {
        "title": "The Union Judiciary — Supreme Court & High Courts",
        "part": "PART V & VI",
        "article_range": "124-147 (SC) & 214-231 (HC)",
        "key_articles": ["124", "129", "131", "136", "141", "142", "214", "226"],
        "overview": (
            "India has a unified, integrated judicial system. The Supreme Court of India (Article 124) is the apex constitutional "
            "court and court of record (Article 129). It exercises original jurisdiction (Article 131), appellate jurisdiction, "
            "Special Leave Petitions (Article 136), advisory jurisdiction (Article 143), and powers to do complete justice (Article 142). "
            "The law declared by the Supreme Court is binding on all courts in India (Article 141). High Courts in the States (Article 214) "
            "exercise extensive writ jurisdiction under Article 226."
        ),
        "keywords": ["judiciary", "supreme court", "high court", "judges", "article 124", "article 226", "article 32", "court of record"]
    },
    "emergency_provisions": {
        "title": "Emergency Provisions (Part XVIII, Articles 352-360)",
        "part": "PART XVIII",
        "article_range": "352 to 360",
        "key_articles": ["352", "356", "360"],
        "overview": (
            "Part XVIII contains emergency powers to safeguard the security and sovereignty of the nation: "
            "(1) National Emergency (Article 352) on grounds of war, external aggression, or armed rebellion; "
            "(2) State Emergency / President's Rule (Article 356) on failure of constitutional machinery in a State; and "
            "(3) Financial Emergency (Article 360). Under modern constitutional law (S.R. Bommai case and 44th Amendment), "
            "emergency proclamations are subject to judicial review, and Articles 20 and 21 cannot be suspended."
        ),
        "keywords": ["emergency", "national emergency", "presidents rule", "article 352", "article 356", "article 360", "part xviii"]
    },
    "amendment_process": {
        "title": "Amendment of the Constitution (Part XX, Article 368)",
        "part": "PART XX",
        "article_range": "368",
        "key_articles": ["368"],
        "overview": (
            "Part XX contains Article 368, which confers power upon Parliament to amend the Constitution by way of addition, "
            "variation or repeal, following a special majority process. Under the landmark Kesavananda Bharati doctrine (1973), "
            "this constituent power is limited: Parliament cannot destroy or alter the Basic Structure of the Constitution."
        ),
        "keywords": ["amendment", "article 368", "basic structure", "special majority", "constituent power"]
    },
    "schedules": {
        "title": "Schedules of the Constitution of India (First to Twelfth Schedule)",
        "part": "Schedules",
        "article_range": "Schedules 1-12",
        "key_articles": ["SCHEDULE_1", "SCHEDULE_2", "SCHEDULE_7", "SCHEDULE_8", "SCHEDULE_10"],
        "overview": (
            "The Constitution contains 12 Schedules that provide detailed administrative lists and provisions: "
            "First Schedule (States and UTs), Second Schedule (Emoluments), Third Schedule (Oaths), "
            "Fourth Schedule (Allocation of Rajya Sabha seats), Fifth & Sixth Schedules (Scheduled Areas and Tribal Areas), "
            "Seventh Schedule (Union, State, and Concurrent Legislative Lists), Eighth Schedule (Recognized Languages), "
            "Ninth Schedule (Validation of certain Acts), Tenth Schedule (Anti-Defection Law), "
            "Eleventh Schedule (Panchayats), and Twelfth Schedule (Municipalities)."
        ),
        "keywords": ["schedules", "seventh schedule", "eighth schedule", "tenth schedule", "anti-defection", "lists"]
    }
}


def clean_constitutional_text(text: str) -> str:
    """Clean PDF extraction noise while preserving constitutional wording."""
    if not text:
        return ""

    # Remove headers such as 'THE CONSTITUTION OF INDIA' or '(Part III.—Fundamental Rights)'
    lines = text.split("\n")
    cleaned_lines = []
    header_regex = re.compile(
        r"^(?:\d+\s+)?(?:THE CONSTITUTION OF INDIA|\(Part\s+[IVXLCDM]+.*?\)|[A-Z\s]{4,}\s+Contents)\s*$",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if header_regex.match(stripped):
            continue
        cleaned_lines.append(stripped)

    joined = " ".join(cleaned_lines)
    # Fix broken hyphenations
    joined = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", joined)
    # Clean redundant whitespace
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def parse_constitution_documents(
    pages_data: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Parse extracted page data into structured constitutional records."""
    articles: List[Dict[str, Any]] = []

    # Map of page_number -> text
    page_map = {p["page_number"]: p["text"] for p in pages_data}

    # 1. Add Preamble
    preamble_page = None
    for p_num, p_text in page_map.items():
        if "PREAMBLE" in p_text.upper() and "WE, THE PEOPLE OF INDIA" in p_text:
            preamble_page = p_num
            break

    if preamble_page:
        preamble_raw = page_map[preamble_page]
        preamble_clean = clean_constitutional_text(preamble_raw)
        # Extract text starting at WE, THE PEOPLE
        idx = preamble_clean.find("WE, THE PEOPLE OF INDIA")
        if idx != -1:
            preamble_clean = preamble_clean[idx:]

        curated = LANDMARK_PROVISIONS.get("PREAMBLE", {})
        articles.append(
            {
                "article_number": "PREAMBLE",
                "article_title": curated.get(
                    "title", "Preamble to the Constitution of India"
                ),
                "part": "Preamble",
                "part_title": "Preamble",
                "page_start": preamble_page,
                "page_end": preamble_page,
                "constitutional_text": preamble_clean,
                "simple_explanation": curated.get("simple_explanation", ""),
                "examples": curated.get("examples", []),
                "key_principles": curated.get("key_principles", []),
                "keywords": curated.get("keywords", []),
                "source_pdf": metadata.get(
                    "source_file_name", "constitution.pdf"
                ),
                "source_hash": metadata.get("file_hash_sha256", ""),
                "status": "verified",
            }
        )

    # 2. Extract Table of Contents entries to know all canonical articles
    toc_articles: Dict[str, str] = {}
    for p_num in range(3, min(32, len(pages_data))):
        txt = page_map.get(p_num, "")
        for line in txt.split("\n"):
            m = re.match(
                r"^\s*\[?(\d+[A-Z]?)\.\s+([^0-9]+?)(?:\.\s*|\?|\.\—|\—|\s*$)",
                line.strip(),
            )
            if m:
                num = m.group(1).upper()
                title = m.group(2).strip(" .[]—\xad")
                if not any(
                    title.startswith(x)
                    for x in ["Subs", "Ins", "Added", "Omitted", "w.e.f"]
                ):
                    if num not in toc_articles:
                        toc_articles[num] = title

    # 3. Locate articles across body pages (pages 33 through 380)
    current_part = "PART I"
    current_part_title = "THE UNION AND ITS TERRITORY"

    part_regex = re.compile(
        r"PART\s+([IVXLCDM]+[A-Z]?)\s*\n\s*([A-Z\s,–—\-]+)", re.MULTILINE
    )

    # Pre-index body page texts
    body_pages = [p for p in pages_data if 33 <= p["page_number"] <= 385]

    for p_info in body_pages:
        p_num = p_info["page_number"]
        p_text = p_info["text"]

        # Track Part headers
        part_match = part_regex.search(p_text)
        if part_match:
            current_part = f"PART {part_match.group(1)}"
            current_part_title = part_match.group(2).strip()

        # Match article definitions on page
        # e.g.: '14. Equality before law. —The State shall not deny...'
        # or '2[21A. Right to education. —The State...'
        art_match_pat = re.compile(
            r"(?:^|\n)\s*"
            r"(?:\d+\[)?\[?"
            r"(\d+[A-Z]?)"
            r"\.\s+"
            r"([A-Z][^\n\r]+?)"
            r"(?:\.|\?)\s*"
            r"(?:[\u2014\xad—\-]|\[|\(1\)|(?=\n)|$)",
            re.MULTILINE,
        )

        for m in art_match_pat.finditer(p_text):
            art_num = m.group(1).upper()
            raw_title = m.group(2).strip()

            # Ignore footnote numbers matching 'Subs. by' or 'Ins. by'
            if any(
                raw_title.startswith(x)
                for x in ["Subs", "Ins", "Added", "Certain words", "Cl. ("]
            ):
                continue

            # Canonical title from TOC or raw_title
            canonical_title = toc_articles.get(art_num, raw_title)
            # Avoid duplicate articles
            if any(a["article_number"] == art_num for a in articles):
                continue

            # Extract article content snippet starting from this match
            start_pos = m.start()
            # Content continues up to next article pattern or bottom of page/next page
            page_slice = p_text[start_pos:]
            # If next article starts on this page, cut before it
            next_m = art_match_pat.search(page_slice, pos=len(m.group(0)))
            if next_m:
                art_content = page_slice[: next_m.start()]
            else:
                art_content = page_slice
                # Append beginning of next page if available and no new article immediately starts
                next_p = page_map.get(p_num + 1, "")
                if next_p and not next_p.strip().startswith(
                    ("PART", "CHAPTER")
                ):
                    next_art_m = art_match_pat.search(next_p)
                    if next_art_m:
                        art_content += "\n" + next_p[: next_art_m.start()]
                    else:
                        art_content += "\n" + next_p[:800]

            clean_text = clean_constitutional_text(art_content)

            # Check if we have curated metadata for this article
            curated = LANDMARK_PROVISIONS.get(art_num, {})

            simple_explanation = curated.get("simple_explanation")
            if not simple_explanation:
                # Generate rule-based grounded plain English summary
                simple_explanation = (
                    f"Article {art_num} of the Constitution of India ({current_part}) provides for '{canonical_title}'. "
                    f"It establishes constitutional provisions regarding {canonical_title.lower()} within the territory of India."
                )

            examples = curated.get("examples", [])
            key_principles = curated.get(
                "key_principles",
                [
                    f"{current_part}: {current_part_title}",
                    canonical_title,
                    f"Article {art_num}",
                ],
            )
            keywords = curated.get(
                "keywords",
                [
                    f"article {art_num.lower()}",
                    canonical_title.lower(),
                    current_part.lower(),
                ],
            )

            articles.append(
                {
                    "article_number": art_num,
                    "article_title": canonical_title,
                    "part": current_part,
                    "part_title": current_part_title,
                    "page_start": p_num,
                    "page_end": p_num,
                    "constitutional_text": clean_text,
                    "simple_explanation": simple_explanation,
                    "examples": examples,
                    "key_principles": key_principles,
                    "keywords": keywords,
                    "source_pdf": metadata.get(
                        "source_file_name", "constitution.pdf"
                    ),
                    "source_hash": metadata.get("file_hash_sha256", ""),
                    "status": "verified",
                }
            )

    # 4. Fill in any remaining curated landmark articles if not caught by page regex
    for art_key, curated_data in LANDMARK_PROVISIONS.items():
        if art_key == "PREAMBLE":
            continue
        if not any(a["article_number"] == art_key for a in articles):
            articles.append(
                {
                    "article_number": art_key,
                    "article_title": curated_data["title"],
                    "part": curated_data["part"],
                    "part_title": curated_data["part_title"],
                    "page_start": 35,
                    "page_end": 45,
                    "constitutional_text": f"Article {art_key}. {curated_data['title']} (See loaded Constitution PDF).",
                    "simple_explanation": curated_data["simple_explanation"],
                    "examples": curated_data["examples"],
                    "key_principles": curated_data["key_principles"],
                    "keywords": curated_data["keywords"],
                    "source_pdf": metadata.get(
                        "source_file_name", "constitution.pdf"
                    ),
                    "source_hash": metadata.get("file_hash_sha256", ""),
                    "status": "curated_grounded",
                }
            )

    logger.info("Successfully parsed %d constitutional articles", len(articles))
    return articles


def save_articles_json(articles: List[Dict[str, Any]], output_path: Path) -> None:
    """Save parsed constitutional articles to articles.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def load_articles_json(articles_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Load previously parsed articles from JSON."""
    if not articles_path.exists():
        return None
    try:
        with open(articles_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load articles JSON: %s", exc)
        return None
