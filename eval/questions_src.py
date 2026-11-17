"""Source of truth for the evaluation question set (written to questions.jsonl).

Each question has:
  id, lang ("en"/"sv"), category, question, answerable (bool),
  gold        provision ids that must be retrieved (article / annex level),
  gold_units  optional finer ids "<provision>#<paragraph>" for paragraph-level scoring,
  answer      short reference answer,
  evidence    substrings that must occur in the gold provisions' English text
              (checked by eval/check_questions.py so the gold labels stay honest),
  version     "current" or "changed": "changed" means the answer differs between
              the AI Act as adopted in 2024 and as amended by Regulation (EU) 2026/1744;
              those questions also carry answer_2024.
Categories: definition, obligation, prohibition, threshold, scope, rights, procedure,
multi_hop (needs two provisions), cross_regulation (AI Act + GDPR), amendment,
unanswerable (not in the corpus), false_premise (question assumes something untrue).
"""

Q = []


def q(id, lang, category, question, gold, answer, evidence=(), gold_units=(), answerable=True, **kw):
    Q.append(dict(id=id, lang=lang, category=category, question=question, answerable=answerable,
                  gold=list(gold), gold_units=list(gold_units), answer=answer, evidence=list(evidence),
                  version=kw.pop("version", "current"), **kw))


# ---------------------------------------------------------------- AI Act, English
q("aia-01", "en", "definition", "How does the AI Act define an 'AI system'?",
  ["AIA:art_3"], "A machine-based system designed to operate with varying levels of autonomy, that may exhibit adaptiveness after deployment, and that infers from its input how to generate outputs such as predictions, content, recommendations or decisions that can influence physical or virtual environments (Art. 3(1)).",
  ["‘AI system’ means a machine-based system"])
q("aia-02", "en", "definition", "Who counts as a 'deployer' under the AI Act?",
  ["AIA:art_3"], "A natural or legal person, public authority, agency or other body using an AI system under its authority, except for personal non-professional use (Art. 3(4)).",
  ["‘deployer’ means"])
q("aia-03", "en", "definition", "What is a 'provider' in the AI Act?",
  ["AIA:art_3"], "A person or body that develops an AI system or GPAI model (or has it developed) and places it on the market or puts it into service under its own name or trademark, paid or free (Art. 3(3)).",
  ["‘provider’ means"])
q("aia-04", "en", "prohibition", "Is social scoring of people by AI systems allowed under the AI Act?",
  ["AIA:art_5"], "No. Art. 5(1)(c) prohibits AI systems that evaluate or classify people based on social behaviour or personal characteristics where the social score leads to detrimental treatment in unrelated contexts or treatment that is unjustified or disproportionate.",
  ["social score leading to"], gold_units=["AIA:art_5#1"])
q("aia-05", "en", "prohibition", "Can an employer use AI to infer the emotions of its employees at work?",
  ["AIA:art_5"], "No, inferring emotions in the workplace and education institutions is prohibited, except where the system is intended for medical or safety reasons (Art. 5(1)(f)).",
  ["to infer emotions of a natural person in the areas of workplace"], gold_units=["AIA:art_5#1"])
q("aia-06", "en", "prohibition", "Is it allowed to build a facial recognition database by scraping facial images from the internet?",
  ["AIA:art_5"], "No. Creating or expanding facial recognition databases through untargeted scraping of facial images from the internet or CCTV footage is prohibited (Art. 5(1)(e)).",
  ["untargeted scraping of facial images"], gold_units=["AIA:art_5#1"])
q("aia-07", "en", "prohibition", "For which purposes may police use real-time remote biometric identification in publicly accessible spaces?",
  ["AIA:art_5"], "Only where strictly necessary for: targeted search for victims of abduction, trafficking or sexual exploitation and missing persons; preventing a specific, substantial and imminent threat to life or a terrorist attack; or locating/identifying suspects of serious offences listed in Annex II punishable by at least four years (Art. 5(1)(h)), subject to the safeguards in Art. 5(2)-(7).",
  ["‘real-time’ remote biometric identification systems in publicly accessible spaces"], gold_units=["AIA:art_5#1"])
q("aia-08", "en", "scope", "Under what two conditions is an AI system that is a safety component of a product considered high-risk?",
  ["AIA:art_6"], "When (a) it is a safety component of a product (or is itself a product) covered by Union harmonisation legislation in Annex I, and (b) that product must undergo a third-party conformity assessment under that legislation (Art. 6(1)).",
  ["both of the following conditions are fulfilled"], gold_units=["AIA:art_6#1"])
q("aia-09", "en", "scope", "When is an AI system listed in Annex III nevertheless not considered high-risk?",
  ["AIA:art_6"], "When it does not pose a significant risk of harm to health, safety or fundamental rights, e.g. it performs a narrow procedural task, improves the result of a completed human activity, detects decision-making patterns without replacing human assessment, or performs a preparatory task. An Annex III system that profiles natural persons is always high-risk (Art. 6(3)).",
  ["narrow procedural task"], gold_units=["AIA:art_6#3"])
q("aia-10", "en", "multi_hop", "Is an AI tool that screens and filters job applications a high-risk AI system?",
  ["AIA:anx_III", "AIA:art_6"], "Yes. AI systems intended for recruitment or selection, in particular to place targeted job ads, analyse and filter applications and evaluate candidates, are listed in Annex III point 4(a), and Annex III systems are high-risk under Art. 6(2) (unless the Art. 6(3) derogation applies).",
  ["to analyse and filter job applications"])
q("aia-11", "en", "obligation", "What must a risk management system for a high-risk AI system include?",
  ["AIA:art_9"], "A continuous iterative process over the whole lifecycle: identifying and analysing known and foreseeable risks, estimating and evaluating risks (including from reasonably foreseeable misuse), evaluating risks from post-market monitoring data, and adopting targeted risk management measures (Art. 9(2)).",
  ["continuous iterative process"])
q("aia-12", "en", "obligation", "What quality requirements apply to training, validation and testing data for high-risk AI systems?",
  ["AIA:art_10"], "Data sets must be subject to appropriate data governance practices and be relevant, sufficiently representative, and to the best extent possible free of errors and complete in view of the intended purpose, with appropriate statistical properties (Art. 10(2)-(4)).",
  ["sufficiently representative"])
q("aia-13", "en", "obligation", "Do high-risk AI systems have to log events automatically?",
  ["AIA:art_12"], "Yes. They must technically allow for the automatic recording of events (logs) over their lifetime, to ensure traceability appropriate to the intended purpose (Art. 12).",
  ["automatic recording of events"])
q("aia-14", "en", "obligation", "What is the purpose of human oversight for high-risk AI systems?",
  ["AIA:art_14"], "To prevent or minimise risks to health, safety or fundamental rights when the system is used as intended or under reasonably foreseeable misuse; systems must be designed so natural persons can effectively oversee them, including to interpret outputs, decide not to use them and interrupt them with a 'stop' button (Art. 14).",
  ["Human oversight shall aim to prevent or minimise the risks"])
q("aia-15", "en", "obligation", "How long must deployers keep the logs generated by a high-risk AI system?",
  ["AIA:art_26"], "For a period appropriate to the intended purpose, of at least six months, unless Union or national law provides otherwise (Art. 26(6)).",
  ["of at least six months"], gold_units=["AIA:art_26#6"])
q("aia-16", "en", "obligation", "Which deployers have to carry out a fundamental rights impact assessment?",
  ["AIA:art_27"], "Deployers that are bodies governed by public law or private entities providing public services, and deployers of Annex III point 5(b) and (c) systems (creditworthiness/credit scoring and life and health insurance risk assessment and pricing), before deploying an Art. 6(2) high-risk system, except for critical infrastructure systems (Annex III point 2) (Art. 27(1)).",
  ["deployers that are bodies governed by public law"], gold_units=["AIA:art_27#1"])
q("aia-17", "en", "obligation", "Must a chatbot tell users that they are talking to an AI?",
  ["AIA:art_50"], "Yes. Providers must design AI systems that interact directly with natural persons so that people are informed they are interacting with an AI system, unless this is obvious to a reasonably well-informed person (Art. 50(1)).",
  ["informed that they are interacting with an AI system"], gold_units=["AIA:art_50#1"])
q("aia-18", "en", "obligation", "What must deployers do when they publish deep fakes?",
  ["AIA:art_50"], "Disclose that the content has been artificially generated or manipulated; for evidently artistic, creative, satirical or fictional works the duty is limited to disclosing the existence of such content in a way that does not hamper the work (Art. 50(4)).",
  ["constituting a deep fake, shall disclose"], gold_units=["AIA:art_50#4"])
q("aia-19", "en", "threshold", "Above what amount of training compute is a general-purpose AI model presumed to have high-impact capabilities?",
  ["AIA:art_51"], "When the cumulative training compute is greater than 10^25 floating point operations (Art. 51(2)).",
  ["greater than 10^25"], gold_units=["AIA:art_51#2"])
q("aia-20", "en", "obligation", "What documentation obligations do providers of general-purpose AI models have?",
  ["AIA:art_53"], "Draw up and keep up to date technical documentation (Annex XI) for the AI Office and authorities, provide information and documentation to downstream providers (Annex XII), put in place a copyright compliance policy, and publish a sufficiently detailed summary of the training content (Art. 53(1)).",
  ["sufficiently detailed summary about the content used for training"])
q("aia-21", "en", "threshold", "What is the maximum fine for using a prohibited AI practice?",
  ["AIA:art_99"], "Up to EUR 35 000 000 or, for undertakings, up to 7 % of total worldwide annual turnover for the preceding financial year, whichever is higher (Art. 99(3)).",
  ["EUR 35 000 000"], gold_units=["AIA:art_99#3"])
q("aia-22", "en", "threshold", "What fine applies for giving misleading information to a notified body under the AI Act?",
  ["AIA:art_99"], "Up to EUR 7 500 000 or up to 1 % of total worldwide annual turnover, whichever is higher (Art. 99(5)).",
  ["EUR 7 500 000"], gold_units=["AIA:art_99#5"])
q("aia-23", "en", "threshold", "How high can fines on providers of general-purpose AI models be?",
  ["AIA:art_101"], "The Commission may fine GPAI model providers up to 3 % of annual worldwide turnover or EUR 15 000 000, whichever is higher (Art. 101(1)).",
  ["fines not exceeding 3 %"])
q("aia-24", "en", "procedure", "Within how many days must a provider report a serious incident caused by a high-risk AI system?",
  ["AIA:art_73"], "Immediately after establishing a causal link (or its reasonable likelihood) and in any event not later than 15 days after becoming aware of the incident; shorter deadlines apply to widespread infringements and deaths (Art. 73(2)-(4)).",
  ["not later than 15 days"], gold_units=["AIA:art_73#2"])
q("aia-25", "en", "scope", "Does the AI Act apply to AI systems used exclusively for military purposes?",
  ["AIA:art_2"], "No. AI systems placed on the market, put into service or used exclusively for military, defence or national security purposes are excluded (Art. 2(3)).",
  ["exclusively for military, defence or national security purposes"], gold_units=["AIA:art_2#3"])
q("aia-26", "en", "scope", "Is an AI model developed only for scientific research covered by the AI Act?",
  ["AIA:art_2"], "No. AI systems or models, including their output, specifically developed and put into service for the sole purpose of scientific research and development are excluded (Art. 2(6)).",
  ["sole purpose of scientific research and development"], gold_units=["AIA:art_2#6"])
q("aia-27", "en", "rights", "Does a person affected by a decision based on a high-risk AI system have a right to an explanation?",
  ["AIA:art_86"], "Yes. A person subject to a deployer's decision based on an Annex III high-risk system (except point 2) that produces legal or similarly significant effects can obtain clear and meaningful explanations of the role of the AI system and the main elements of the decision (Art. 86(1)).",
  ["clear and meaningful explanations"])
q("aia-28", "en", "multi_hop", "What information must the technical documentation of a high-risk AI system contain?",
  ["AIA:art_11", "AIA:anx_IV"], "At a minimum the elements in Annex IV, e.g. a general description, a detailed description of the elements and development process, monitoring/functioning/control information, performance metrics, the risk management system, changes, standards applied, the EU declaration of conformity and the post-market monitoring plan (Art. 11(1) and Annex IV).",
  ["set out in Annex IV"])
q("aia-29", "en", "multi_hop", "Which conformity assessment procedure applies to a high-risk biometric AI system when the provider has applied harmonised standards?",
  ["AIA:art_43"], "The provider can choose between internal control (Annex VI) and assessment of the quality management system and technical documentation with a notified body (Annex VII) (Art. 43(1)).",
  ["listed in point 1 of Annex III"], gold_units=["AIA:art_43#1"])
q("aia-30", "en", "multi_hop", "What does a provider of a high-risk AI system have to register in the EU database?",
  ["AIA:art_49", "AIA:anx_VIII"], "Before placing on the market, the provider registers itself and the system in the EU database (Art. 49(1)), with the information listed in Annex VIII Section A (e.g. provider details, trade name, intended purpose, status, certificates, member states, instructions for use URL).",
  ["register themselves and their system in the EU database"])
q("aia-31", "en", "obligation", "What must the post-market monitoring system of a high-risk AI system do?",
  ["AIA:art_72"], "Actively and systematically collect, document and analyse relevant data on the system's performance throughout its lifetime, allowing the provider to evaluate continuous compliance, based on a post-market monitoring plan (Art. 72).",
  ["post-market monitoring system"])
q("aia-32", "en", "obligation", "What are the main obligations of providers of high-risk AI systems?",
  ["AIA:art_16"], "Ensure compliance with the Section 2 requirements, indicate their name and address, have a quality management system (Art. 17), keep documentation and logs, undergo conformity assessment, draw up an EU declaration of conformity, affix CE marking, register, take corrective action and demonstrate conformity on request, and meet accessibility requirements (Art. 16).",
  ["Providers of high-risk AI systems shall"])

# ---------------------------------------------------------------- GDPR, English
q("gdpr-01", "en", "procedure", "How quickly must a controller notify a personal data breach to the supervisory authority?",
  ["GDPR:art_33"], "Without undue delay and, where feasible, not later than 72 hours after becoming aware of it, unless the breach is unlikely to result in a risk; late notifications must give reasons for the delay (Art. 33(1)).",
  ["not later than 72 hours"], gold_units=["GDPR:art_33#1"])
q("gdpr-02", "en", "procedure", "When must data subjects be told about a personal data breach?",
  ["GDPR:art_34"], "When the breach is likely to result in a high risk to their rights and freedoms, without undue delay, unless e.g. the data was encrypted, subsequent measures remove the high risk, or individual communication would involve disproportionate effort (Art. 34).",
  ["likely to result in a high risk"])
q("gdpr-03", "en", "threshold", "What is the maximum GDPR fine for violating the basic principles of processing?",
  ["GDPR:art_83"], "Up to EUR 20 000 000 or, for undertakings, up to 4 % of total worldwide annual turnover of the preceding financial year, whichever is higher (Art. 83(5)).",
  ["20 000 000 EUR"], gold_units=["GDPR:art_83#5"])
q("gdpr-04", "en", "threshold", "What fine applies to controllers who breach their obligations on data protection by design or on records of processing?",
  ["GDPR:art_83"], "The lower tier: up to EUR 10 000 000 or 2 % of worldwide annual turnover, whichever is higher, covers controller/processor obligations under Arts 8, 11, 25 to 39, 42 and 43 (Art. 83(4)).",
  ["10 000 000 EUR"], gold_units=["GDPR:art_83#4"])
q("gdpr-05", "en", "rights", "On what grounds can a data subject ask for their personal data to be erased?",
  ["GDPR:art_17"], "When the data are no longer necessary, consent is withdrawn and there is no other legal ground, the data subject objects under Art. 21, the data were unlawfully processed, erasure is required by law, or the data were collected in relation to information society services offered to a child (Art. 17(1)).",
  ["right to obtain from the controller the erasure"])
q("gdpr-06", "en", "obligation", "When is it mandatory to designate a data protection officer?",
  ["GDPR:art_37"], "When processing is carried out by a public authority or body (except courts acting judicially), when core activities require regular and systematic monitoring of data subjects on a large scale, or when core activities consist of large-scale processing of special categories or criminal data (Art. 37(1)).",
  ["shall designate a data protection officer"])
q("gdpr-07", "en", "obligation", "When is a data protection impact assessment required?",
  ["GDPR:art_35"], "When processing, in particular using new technologies, is likely to result in a high risk; in particular for systematic and extensive automated evaluation including profiling with legal or similar effects, large-scale processing of special categories or criminal data, and systematic large-scale monitoring of publicly accessible areas (Art. 35(1),(3)).",
  ["likely to result in a high risk to the rights and freedoms"])
q("gdpr-08", "en", "threshold", "From what age can a child consent on their own to the processing of their data by an online service?",
  ["GDPR:art_8"], "16 years; below that, consent must be given or authorised by the holder of parental responsibility. Member States may set a lower age, not below 13 years (Art. 8(1)).",
  ["at least 16 years old"], gold_units=["GDPR:art_8#1"])
q("gdpr-09", "en", "definition", "What are the lawful bases for processing personal data?",
  ["GDPR:art_6"], "Consent, performance of a contract, legal obligation, vital interests, public interest/official authority, and legitimate interests (Art. 6(1)(a)-(f)).",
  ["Processing shall be lawful only if and to the extent"])
q("gdpr-10", "en", "definition", "Which categories of personal data count as special categories?",
  ["GDPR:art_9"], "Data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, trade union membership, genetic data, biometric data for unique identification, health data, and data on sex life or sexual orientation (Art. 9(1)).",
  ["revealing racial or ethnic origin"])
q("gdpr-11", "en", "rights", "Can a person refuse to be subject to a decision made solely by automated processing?",
  ["GDPR:art_22"], "Yes, a data subject has the right not to be subject to a decision based solely on automated processing, including profiling, that produces legal or similarly significant effects, unless necessary for a contract, authorised by law, or based on explicit consent, with safeguards such as human intervention (Art. 22).",
  ["based solely on automated processing"])
q("gdpr-12", "en", "rights", "What does the right to data portability give a data subject?",
  ["GDPR:art_20"], "The right to receive their personal data in a structured, commonly used and machine-readable format and transmit it to another controller, where processing is based on consent or contract and carried out by automated means (Art. 20).",
  ["structured, commonly used and machine-readable format"])
q("gdpr-13", "en", "procedure", "How long does a controller have to respond to a data subject's access request?",
  ["GDPR:art_12"], "Without undue delay and in any event within one month of receipt; extendable by two further months for complex or numerous requests, with the data subject informed within the first month (Art. 12(3)).",
  ["within one month of receipt of the request"], gold_units=["GDPR:art_12#3"])
q("gdpr-14", "en", "obligation", "Do small companies need to keep records of processing activities?",
  ["GDPR:art_30"], "Organisations with fewer than 250 employees are exempt unless the processing is likely to result in a risk, is not occasional, or includes special categories or criminal data (Art. 30(5)).",
  ["fewer than 250 persons"], gold_units=["GDPR:art_30#5"])
q("gdpr-15", "en", "obligation", "What does data protection by design and by default require?",
  ["GDPR:art_25"], "Implementing appropriate technical and organisational measures, such as pseudonymisation and data minimisation, both when determining the means and at the time of processing, and ensuring that by default only data necessary for each specific purpose are processed (Art. 25).",
  ["Data protection by design and by default"])
q("gdpr-16", "en", "definition", "What is 'pseudonymisation' under the GDPR?",
  ["GDPR:art_4"], "Processing personal data so it can no longer be attributed to a specific data subject without additional information, which is kept separately and subject to technical and organisational measures (Art. 4(5)).",
  ["‘pseudonymisation’ means"])
q("gdpr-17", "en", "rights", "Can consent to processing be withdrawn, and how easy must that be?",
  ["GDPR:art_7"], "Yes, at any time; withdrawal does not affect the lawfulness of earlier processing, and it must be as easy to withdraw as to give consent (Art. 7(3)).",
  ["as easy to withdraw as to give consent"])
q("gdpr-18", "en", "obligation", "What must a contract between a controller and a processor contain?",
  ["GDPR:art_28"], "It must set out subject matter, duration, nature and purpose, data types and categories of data subjects, and require the processor to act only on documented instructions, ensure confidentiality, security, sub-processor rules, assistance, deletion/return of data and audits (Art. 28(3)).",
  ["governed by a contract or other legal act"])
q("gdpr-19", "en", "procedure", "Can personal data be transferred to a third country that has an adequacy decision?",
  ["GDPR:art_45"], "Yes. A transfer may take place where the Commission has decided that the third country ensures an adequate level of protection; no specific authorisation is required (Art. 45(1)).",
  ["adequate level of protection"])

# ---------------------------------------------------------------- Cross-regulation
q("x-01", "en", "cross_regulation", "Does the AI Act change or override the GDPR?",
  ["AIA:art_2"], "No. Union data protection law applies to personal data processed under the AI Act, and the AI Act does not affect the GDPR, without prejudice to Articles 4a and 59 of the AI Act (Art. 2(7)).",
  ["shall not affect Regulation (EU) 2016/679"], gold_units=["AIA:art_2#7"])
q("x-02", "en", "cross_regulation", "If I have already done a DPIA under the GDPR, do I still need a fundamental rights impact assessment under the AI Act?",
  ["AIA:art_27", "GDPR:art_35"], "The FRIA is still required, but where obligations are already met by the DPIA under GDPR Art. 35, the deployer may cross-reference the relevant DPIA sections or include parts of it in the FRIA (AI Act Art. 27(4)).",
  ["data protection impact assessment conducted pursuant to Article 35"], gold_units=["AIA:art_27#4"])
q("x-03", "en", "cross_regulation", "May an AI provider process sensitive personal data such as ethnicity to detect bias in a high-risk AI system?",
  ["AIA:art_4a", "GDPR:art_9"], "Exceptionally, yes, to the extent strictly necessary for bias detection and correction, if conditions are met: no alternative (e.g. synthetic or anonymised data), technical limits on re-use and privacy-preserving measures, strict access controls, no transfer to third parties, deletion once bias is corrected, and records of processing (AI Act Art. 4a). These are special categories under GDPR Art. 9(1).",
  ["bias detection and correction"])
q("x-04", "en", "cross_regulation", "A bank uses an AI credit scoring system to reject loan applications automatically. Which rules from the AI Act and GDPR apply?",
  ["AIA:anx_III", "GDPR:art_22"], "Credit scoring of natural persons is high-risk under Annex III point 5(b) of the AI Act (with FRIA and explanation rights), and a solely automated decision with legal effects falls under GDPR Art. 22, requiring a legal basis such as contract or explicit consent plus safeguards including human intervention.",
  ["creditworthiness of natural persons"])
q("x-05", "en", "cross_regulation", "How do the AI Act and the GDPR define biometric data?",
  ["AIA:art_3", "GDPR:art_4"], "Both define it as personal data resulting from specific technical processing relating to physical, physiological or behavioural characteristics of a natural person (GDPR adds 'which allow or confirm the unique identification'). AI Act Art. 3(34); GDPR Art. 4(14).",
  ["‘biometric data’ means personal data resulting from specific technical processing"])

# ---------------------------------------------------------------- Amendments (Digital Omnibus on AI, Regulation (EU) 2026/1744)
q("amd-01", "en", "amendment", "From when do the obligations for high-risk AI systems listed in Annex III apply?",
  ["AIA:art_113"], "From 2 December 2027 (Art. 113(c)(i), as amended by Regulation (EU) 2026/1744).",
  ["2 December 2027"], version="changed",
  answer_2024="From 2 August 2026, the general date of application (Art. 113 as adopted).")
q("amd-02", "en", "amendment", "When do high-risk requirements apply to AI systems that are safety components of products under Annex I?",
  ["AIA:art_113"], "From 2 August 2028 (Art. 113(c)(ii), as amended).",
  ["2 August 2028"], version="changed",
  answer_2024="From 2 August 2027 (Art. 113(c) as adopted).")
q("amd-03", "en", "amendment", "What exactly does the AI literacy obligation require from providers and deployers?",
  ["AIA:art_4"], "To take measures to support the development of AI literacy of staff and others operating AI systems on their behalf; it does not require guaranteeing any specific level of AI literacy of an individual. The Commission and Member States support SMEs in complying (Art. 4(1)-(2), as amended).",
  ["support the development of AI literacy"], gold_units=["AIA:art_4#1"], version="changed",
  answer_2024="To take measures to ensure, to their best extent, a sufficient level of AI literacy of their staff (Art. 4 as adopted).")
q("amd-04", "en", "amendment", "By when must each Member State have an AI regulatory sandbox operational?",
  ["AIA:art_57"], "By 2 August 2027 (Art. 57(1), as amended).",
  ["operational by 2 August 2027"], gold_units=["AIA:art_57#1"], version="changed",
  answer_2024="By 2 August 2026 (Art. 57(1) as adopted).")
q("amd-05", "en", "amendment", "Does the AI Act ban apps that generate sexually explicit deepfakes of real people without consent?",
  ["AIA:art_5", "AIA:art_113"], "Yes. Art. 5(1)(ba), inserted in 2026, prohibits AI systems that generate or manipulate realistic images, videos or audio of an identifiable person's intimate parts or sexual activity without their explicit consent; this applies from 2 December 2026 (Art. 113(a)).",
  ["identifiable natural person’s intimate parts"], gold_units=["AIA:art_5#1"], version="changed",
  answer_2024="Not as a specific prohibition; such systems were only covered by the deep fake transparency duty in Art. 50(4).")
q("amd-06", "en", "amendment", "Which companies can comply with the quality management system requirements in a simplified way?",
  ["AIA:art_63"], "SMEs, including start-ups, without partner or linked enterprises (Art. 63(1), as amended).",
  ["SMEs, including start-ups, may comply with certain elements of the quality management system"], gold_units=["AIA:art_63#1"], version="changed",
  answer_2024="Only microenterprises within the meaning of Recommendation 2003/361/EC (Art. 63(1) as adopted).")
q("amd-07", "en", "amendment", "By when must providers of generative AI systems already on the market before August 2026 comply with the machine-readable marking obligation?",
  ["AIA:art_111", "AIA:art_50"], "By 2 December 2026, for systems placed on the market before 2 August 2026 (Art. 111(4) referring to Art. 50(2)).",
  ["comply with Article 50(2) by 2 December 2026"], gold_units=["AIA:art_111#4"], version="changed",
  answer_2024="No transitional period: Art. 50 applied from 2 August 2026.")
q("amd-08", "en", "amendment", "How are fines under the AI Act capped for small mid-cap enterprises (SMCs)?",
  ["AIA:art_99"], "For SMCs, each fine under Art. 99(4) and (5) is up to the percentage or amount set there, whichever is lower (Art. 99(6a)).",
  ["In the case of SMCs"], gold_units=["AIA:art_99#6a"], version="changed",
  answer_2024="The 2024 text had no SMC category; only SMEs and start-ups got the 'whichever is lower' rule (Art. 99(6)).")
q("amd-09", "en", "amendment", "Can a fundamental rights impact assessment reuse parts of a GDPR data protection impact assessment?",
  ["AIA:art_27"], "Yes. The deployer may include cross-references to relevant sections of the DPIA or include relevant parts of it in the FRIA, and the AI Office questionnaire template supports this (Art. 27(4)-(5), as amended).",
  ["include cross-references to the relevant sections"], gold_units=["AIA:art_27#4"], version="changed",
  answer_2024="The FRIA 'shall complement' the DPIA (Art. 27(4) as adopted), without explicit cross-referencing.")
q("amd-10", "en", "amendment", "From when are the new prohibitions on AI-generated intimate images and child sexual abuse material applicable?",
  ["AIA:art_113"], "From 2 December 2026 (Art. 113(a): Art. 5(1) points (ba) and (bb) and Art. 5(1a)-(1b)).",
  ["shall apply from 2 December 2026"], version="changed",
  answer_2024="These prohibitions did not exist in the 2024 text.")

# ---------------------------------------------------------------- Swedish questions (cross-lingual)
q("sv-01", "sv", "procedure", "Inom hur många timmar ska en personuppgiftsincident anmälas till tillsynsmyndigheten?",
  ["GDPR:art_33"], "Utan onödigt dröjsmål och om möjligt inom 72 timmar efter att den personuppgiftsansvarige fått vetskap om den (artikel 33.1).",
  ["not later than 72 hours"], gold_units=["GDPR:art_33#1"], evidence_sv=["72 timmar"])
q("sv-02", "sv", "threshold", "Vilken är den högsta sanktionsavgiften för förbjudna AI-metoder enligt AI-förordningen?",
  ["AIA:art_99"], "Upp till 35 000 000 euro eller 7 % av den totala globala årsomsättningen, beroende på vilket belopp som är högst (artikel 99.3).",
  ["EUR 35 000 000"], gold_units=["AIA:art_99#3"], evidence_sv=["35 000 000"])
q("sv-03", "sv", "prohibition", "Får en arbetsgivare använda AI för att avläsa anställdas känslor på arbetsplatsen?",
  ["AIA:art_5"], "Nej, det är förbjudet utom när systemet används av medicinska skäl eller säkerhetsskäl (artikel 5.1 f).",
  ["to infer emotions of a natural person in the areas of workplace"], gold_units=["AIA:art_5#1"], evidence_sv=["arbetsplatsen"])
q("sv-04", "sv", "amendment", "När börjar kraven för AI-system med hög risk enligt bilaga III att tillämpas?",
  ["AIA:art_113"], "Från och med den 2 december 2027 (artikel 113 c i, i dess ändrade lydelse).",
  ["2 December 2027"], version="changed", evidence_sv=["2 december 2027"],
  answer_2024="Från och med den 2 augusti 2026.")
q("sv-05", "sv", "threshold", "Från vilken ålder kan ett barn själv samtycka till behandling av personuppgifter i en onlinetjänst?",
  ["GDPR:art_8"], "16 år; medlemsstaterna får sänka åldern, dock inte under 13 år (artikel 8.1).",
  ["at least 16 years old"], gold_units=["GDPR:art_8#1"], evidence_sv=["16 år"])
q("sv-06", "sv", "definition", "Vad är en leverantör enligt AI-förordningen?",
  ["AIA:art_3"], "En fysisk eller juridisk person, myndighet, byrå eller annat organ som utvecklar ett AI-system eller en AI-modell för allmänna ändamål, eller låter utveckla det, och släpper ut det på marknaden eller tar det i bruk i eget namn eller under eget varumärke (artikel 3.3).",
  ["‘provider’ means"], evidence_sv=["leverantör"])
q("sv-07", "sv", "obligation", "Måste en chattbot tala om för användaren att den är en AI?",
  ["AIA:art_50"], "Ja, leverantörer ska se till att fysiska personer informeras om att de interagerar med ett AI-system, om det inte är uppenbart (artikel 50.1).",
  ["informed that they are interacting with an AI system"], gold_units=["AIA:art_50#1"], evidence_sv=["interagerar med ett AI-system"])
q("sv-08", "sv", "threshold", "Vid vilken beräkningsmängd antas en AI-modell för allmänna ändamål ha kapacitet med hög påverkansgrad?",
  ["AIA:art_51"], "När den kumulativa beräkningsmängden för träningen överstiger 10^25 flyttalsberäkningar (artikel 51.2).",
  ["greater than 10^25"], gold_units=["AIA:art_51#2"], evidence_sv=["flyttalsberäkningar"])
q("sv-09", "sv", "rights", "Vilka skäl ger en registrerad rätt att få sina personuppgifter raderade?",
  ["GDPR:art_17"], "Bland annat när uppgifterna inte längre behövs, samtycket återkallas, den registrerade invänder, uppgifterna behandlats olagligt eller radering krävs enligt lag (artikel 17.1).",
  ["right to obtain from the controller the erasure"], evidence_sv=["raderade"])
q("sv-10", "sv", "obligation", "När måste en konsekvensbedömning avseende dataskydd göras?",
  ["GDPR:art_35"], "När en behandling sannolikt leder till en hög risk för fysiska personers rättigheter och friheter, särskilt vid profilering med rättsliga följder, storskalig behandling av känsliga uppgifter eller storskalig övervakning av allmän plats (artikel 35).",
  ["likely to result in a high risk to the rights and freedoms"], evidence_sv=["konsekvensbedömning"])
q("sv-11", "sv", "obligation", "Måste en myndighet utse ett dataskyddsombud?",
  ["GDPR:art_37"], "Ja, när behandlingen utförs av en myndighet eller ett offentligt organ, utom domstolar i deras dömande verksamhet (artikel 37.1 a).",
  ["shall designate a data protection officer"], evidence_sv=["dataskyddsombud"])
q("sv-12", "sv", "multi_hop", "Är ett AI-system som sorterar jobbansökningar ett AI-system med hög risk?",
  ["AIA:anx_III", "AIA:art_6"], "Ja, AI-system för rekrytering och urval, t.ex. för att analysera och filtrera ansökningar, anges i bilaga III punkt 4 a och är därmed AI-system med hög risk enligt artikel 6.2.",
  ["to analyse and filter job applications"], evidence_sv=["rekrytering"])
q("sv-13", "sv", "obligation", "Hur länge ska tillhandahållare spara loggar från ett AI-system med hög risk?",
  ["AIA:art_26"], "Under en period som är lämplig för systemets avsedda ändamål, minst sex månader, om inte annat föreskrivs i unionsrätten eller nationell rätt (artikel 26.6).",
  ["of at least six months"], gold_units=["AIA:art_26#6"], evidence_sv=["sex månader"])
q("sv-14", "sv", "amendment", "Vad kräver skyldigheten om AI-kunnighet av leverantörer och tillhandahållare?",
  ["AIA:art_4"], "De ska vidta åtgärder för att stödja utvecklingen av AI-kunnighet hos sin personal; det krävs inte att en viss nivå garanteras för enskilda personer (artikel 4.1, i dess ändrade lydelse).",
  ["support the development of AI literacy"], gold_units=["AIA:art_4#1"], version="changed", evidence_sv=["AI-kunnighet"],
  answer_2024="Att i största möjliga utsträckning säkerställa en tillräcklig nivå av AI-kunnighet hos personalen.")

# ---------------------------------------------------------------- Unanswerable from the corpus
UNANSWERABLE_EN = [
    ("un-01", "Which Swedish authority acts as market surveillance authority for the AI Act in Sweden?"),
    ("un-02", "What does the Digital Services Act require online platforms to do about illegal content?"),
    ("un-03", "How much does a conformity assessment by a notified body typically cost?"),
    ("un-04", "What did the Court of Justice of the EU decide in the Schrems II judgment?"),
    ("un-05", "What rights do consumers have to opt out of the sale of their data under the California Consumer Privacy Act?"),
    ("un-06", "What is the maximum fine under the UK GDPR after Brexit?"),
    ("un-07", "Which companies have been fined under the AI Act so far?"),
    ("un-08", "Which harmonised standards for the AI Act have been published by CEN-CENELEC?"),
    ("un-09", "What is the corporate income tax rate for AI start-ups in the EU?"),
]
for id_, text in UNANSWERABLE_EN:
    q(id_, "en", "unanswerable", text, [], "The AI Act and GDPR texts do not answer this; the assistant should say so.",
      answerable=False)
q("un-10", "sv", "unanswerable", "Vilken svensk myndighet ansvarar för tillsynen av AI-förordningen i Sverige?", [],
  "Det framgår inte av AI-förordningen eller dataskyddsförordningen; assistenten ska säga att svaret saknas.",
  answerable=False)
q("un-11", "sv", "unanswerable", "Hur hög är sanktionsavgiften i kronor som Integritetsskyddsmyndigheten gav Spotify?", [],
  "Det framgår inte av texterna; assistenten ska säga att svaret saknas.", answerable=False)

# ---------------------------------------------------------------- False premise
q("fp-01", "en", "false_premise", "What is the training-compute threshold above which an AI system becomes high-risk under the AI Act?",
  ["AIA:art_6", "AIA:art_51"], "There is no compute threshold for high-risk AI systems; high-risk status follows Art. 6 and Annexes I/III. The 10^25 FLOP threshold in Art. 51(2) concerns general-purpose AI models with systemic risk.",
  ["greater than 10^25"], answerable=False)
q("fp-02", "en", "false_premise", "Which article of the GDPR requires a fundamental rights impact assessment for AI systems?",
  ["AIA:art_27", "GDPR:art_35"], "None. The GDPR requires a data protection impact assessment (Art. 35); the fundamental rights impact assessment comes from AI Act Art. 27.",
  ["fundamental rights impact assessment"], answerable=False)
q("fp-03", "en", "false_premise", "Since the AI Act bans all biometric identification, can shops still use facial recognition for security?",
  ["AIA:art_5", "AIA:anx_III"], "The premise is wrong: the AI Act does not ban all biometric identification. Art. 5(1)(h) prohibits real-time remote biometric identification in public spaces for law enforcement, with exceptions; other remote biometric identification systems are high-risk under Annex III point 1(a), and GDPR Art. 9 still governs biometric data.",
  ["remote biometric identification systems"], answerable=False)


# ---------------------------------------------------------------- answer keys
# Automatic correctness check for generated answers: every group must match
# (any alternative in a group, case-insensitive regex). Kept to the decisive
# facts (numbers, dates, yes/no, named concepts) so paraphrases still pass.
# keys_2024 detect an outdated answer that matches the pre-2026 text.
KEYS = {
    "aia-01": [[r"machine-based"], [r"autonom"], [r"infer"]],
    "aia-02": [[r"under its authority|using an AI system"], [r"personal non-professional|non-professional"]],
    "aia-03": [[r"develops|developed"], [r"own name|trademark|places? (it )?on the market"]],
    "aia-04": [[r"\bno\b|prohibit|not allowed|banned"]],
    "aia-05": [[r"\bno\b|prohibit|not allowed|banned"], [r"medical|safety"]],
    "aia-06": [[r"\bno\b|prohibit|not allowed|banned"], [r"untargeted|scrap"]],
    "aia-07": [[r"victim|abduction|missing"], [r"threat|terrorist"], [r"suspect|criminal offence|offence"]],
    "aia-08": [[r"Annex I\b|harmonisation legislation"], [r"third-party conformity assessment|third party"]],
    "aia-09": [[r"significant risk"], [r"narrow procedural|preparatory|previously completed|patterns"]],
    "aia-10": [[r"\byes\b|is (a )?high-risk|considered high-risk|classified as high-risk|qualif"], [r"Annex III"]],
    "aia-11": [[r"identif"], [r"risk"], [r"measure"]],
    "aia-12": [[r"representative"], [r"error|complete|relevant"]],
    "aia-13": [[r"\byes\b|automatic"], [r"log|record"]],
    "aia-14": [[r"health|safety|fundamental rights"], [r"interrupt|stop|override|disregard|intervene"]],
    "aia-15": [[r"six months|6 months"]],
    "aia-16": [[r"public law|public services|public"], [r"credit|insurance|5 ?\(b\)|5\(c\)"]],
    "aia-17": [[r"\byes\b|must|shall|inform"], [r"interact"]],
    "aia-18": [[r"disclose|label|inform"], [r"artificially generated|manipulated|AI-generated"]],
    "aia-19": [[r"10\^?\s?25|10²⁵|1e25"]],
    "aia-20": [[r"technical documentation"], [r"summary|copyright|downstream"]],
    "aia-21": [[r"35 ?000 ?000|35 million|35m"], [r"7 ?%|7 per ?cent"]],
    "aia-22": [[r"7 ?500 ?000|7[.,]5 million"], [r"1 ?%|1 per ?cent"]],
    "aia-23": [[r"15 ?000 ?000|15 million"], [r"3 ?%|3 per ?cent"]],
    "aia-24": [[r"15 days|fifteen days"]],
    "aia-25": [[r"\bno\b|does not apply|not apply|exclud"], [r"military|defence|national security"]],
    "aia-26": [[r"\bno\b|does not apply|not apply|exclud|not covered"], [r"research"]],
    "aia-27": [[r"\byes\b|right"], [r"explanation"]],
    "aia-28": [[r"Annex IV"]],
    "aia-29": [[r"internal control|Annex VI"], [r"quality management|notified body|Annex VII"]],
    "aia-30": [[r"EU database|database"]],
    "aia-31": [[r"collect|document|analy"], [r"performance|lifetime|compliance"]],
    "aia-32": [[r"quality management"], [r"conformity assessment|CE marking|declaration of conformity"]],
    "gdpr-01": [[r"72 hours|72 timmar"]],
    "gdpr-02": [[r"high risk"]],
    "gdpr-03": [[r"20 ?000 ?000|20 million"], [r"4 ?%|4 per ?cent"]],
    "gdpr-04": [[r"10 ?000 ?000|10 million"], [r"2 ?%|2 per ?cent"]],
    "gdpr-05": [[r"no longer necessary"], [r"withdraw|unlawful|object"]],
    "gdpr-06": [[r"public authority|public bod"], [r"large scale|large-scale"]],
    "gdpr-07": [[r"high risk"]],
    "gdpr-08": [[r"\b16\b"], [r"\b13\b|parental"]],
    "gdpr-09": [[r"consent"], [r"contract"], [r"legitimate interest"]],
    "gdpr-10": [[r"racial|ethnic"], [r"health"], [r"biometric|genetic"]],
    "gdpr-11": [[r"\byes\b|right not to be subject|right"], [r"solely"]],
    "gdpr-12": [[r"machine-readable|structured"], [r"transmit|another controller"]],
    "gdpr-13": [[r"one month|1 month"], [r"two (further )?months|2 (further )?months|extend"]],
    "gdpr-14": [[r"250"]],
    "gdpr-15": [[r"technical and organisational|technical and organizational"], [r"by default|necessary"]],
    "gdpr-16": [[r"additional information"], [r"separately|attributed"]],
    "gdpr-17": [[r"\byes\b|any time|withdraw"], [r"as easy"]],
    "gdpr-18": [[r"documented instructions|instructions"], [r"confidential|security|sub-?processor"]],
    "gdpr-19": [[r"\byes\b|may take place|can be transferred|permitted|allowed"], [r"adequa"]],
    "x-01": [[r"\bno\b|does not affect|not affect|without prejudice"]],
    "x-02": [[r"\byes\b|still|required"], [r"cross-reference|refer|include|complement|reuse|re-use"]],
    "x-03": [[r"\byes\b|exceptionally|may|can"], [r"strictly necessary|safeguard|conditions"]],
    "x-04": [[r"high-risk|Annex III"], [r"Article 22|automated"]],
    "x-05": [[r"physical|physiological|behavioural|behavioral"]],
    "amd-01": [[r"2 December 2027|December 2027"]],
    "amd-02": [[r"2 August 2028|August 2028"]],
    "amd-03": [[r"support"], [r"not require|does not require|no specific level|guarantee"]],
    "amd-04": [[r"2 August 2027|August 2027"]],
    "amd-05": [[r"\byes\b|prohibit|banned"], [r"consent"]],
    "amd-06": [[r"SME|small and medium|start-up"]],
    "amd-07": [[r"2 December 2026|December 2026"]],
    "amd-08": [[r"lower"]],
    "amd-09": [[r"\byes\b|may|can"], [r"cross-reference|include"]],
    "amd-10": [[r"2 December 2026|December 2026"]],
    "sv-01": [[r"72 timmar|72 hours|72"]],
    "sv-02": [[r"35 ?000 ?000|35 miljoner"], [r"7 ?%|7 procent"]],
    "sv-03": [[r"\bnej\b|förbjud|inte tillåt"], [r"medicinsk|säkerhet"]],
    "sv-04": [[r"2 december 2027|december 2027"]],
    "sv-05": [[r"\b16\b"], [r"\b13\b|föräld|vårdnad"]],
    "sv-06": [[r"utvecklar|utveckla"], [r"eget namn|varumärke|släpper ut"]],
    "sv-07": [[r"\bja\b|måste|ska|informer"]],
    "sv-08": [[r"10\^?\s?25|10²⁵"]],
    "sv-09": [[r"inte längre|nödvändig|behövs"], [r"återkall|olag|invänd"]],
    "sv-10": [[r"hög risk"]],
    "sv-11": [[r"\bja\b|ska|måste"], [r"myndighet|offentlig"]],
    "sv-12": [[r"\bja\b|hög risk"], [r"bilaga III|bilaga 3"]],
    "sv-13": [[r"sex månader|6 månader"]],
    "sv-14": [[r"stödja|främja"]],
    "fp-01": [[r"no (such )?(compute )?threshold|does not (set|define|contain|establish)|not based on (training )?compute|there is no|isn't a|is not a"], [r"general-purpose|GPAI|systemic"]],
    "fp-02": [[r"none|no article|does not|doesn't|not (in|part of) the GDPR|AI Act"], [r"Article 27|AI Act"]],
    "fp-03": [[r"not (ban|prohibit) all|does not ban all|not all|incorrect|premise|only prohibits|specific|not a blanket"]],
}
KEYS_2024 = {
    "amd-01": [[r"2 August 2026"]],
    "amd-02": [[r"2 August 2027"]],
    "amd-03": [[r"sufficient level"]],
    "amd-04": [[r"2 August 2026"]],
    "amd-06": [[r"microenterprise|micro-enterprise"]],
    "sv-04": [[r"2 augusti 2026"]],
    "sv-14": [[r"tillräcklig nivå"]],
}
for _x in Q:
    _x["keys"] = KEYS.get(_x["id"], [])
    if _x["id"] in KEYS_2024:
        _x["keys_2024"] = KEYS_2024[_x["id"]]


if __name__ == "__main__":
    import json
    from pathlib import Path

    out = Path(__file__).with_name("questions.jsonl")
    ids = [x["id"] for x in Q]
    assert len(ids) == len(set(ids)), "duplicate ids"
    with out.open("w", encoding="utf-8") as f:
        for x in Q:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"wrote {len(Q)} questions to {out}")
