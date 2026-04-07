Mailurile primite pana in acest moment au fost in romana
asa ca voi raspunde la intrebari in romana.
notita pt rulare schimbati din run_querries queriurile introduse
OPENAI_API_KEY=XXX
^ asta este intr un .env file pentru rulare

3.1 Approach

Describe your system architecture.

    What components does it include?
Avem 4 faze diferite acestea includ:
faza 1 care preia query-ul si din fiecare cuvant incearca sa extraga coloana din care vine de exemplu daca avem ... din romania bucata cu din romania va fi pusa ca un filtru la country code RO.
NOTITA:aici ar fi ajutat foarte mult popularea tabelelor care nu au valori de exemplu daca o la o tara country codeul este null dar putem extrage din latitudine longitudine ca este din romania vom sari total peste companie chiar daca este buna.
faza 2 evalueaza companiile si incearca sa puna restrictiile necesare extrase din queryu initial. Pentru fiecare companie trimite datele companiei si constrangerile la GPT si cere true/false daca se potriveste. Implementeaza si un circuit breaker daca LLM-ul esueaza de 3 ori sau ajunge la quota API, sare direct la keyword matching. Returneaza companii cu score de potrivire 0-1.
faza 3 extrage meaningul din query si descrierile companiilor folosind sentence-transformers (all-MiniLM-L6-v2). Converteste textul in vectori de 384 numere si calculeaza similaritate cosinus intre query si fiecare companie. Score semantic inalt 0.9 inseamna ca descrierea companiei are acelasi sens ca query-ul chiar daca cuvintele sunt diferite.
faza 4 combina score-urile din faza 2 (60%) si faza 3 (40%) Calculeaza confidence score final pe baza completitudinii datelor companiei si returneaza lista finala de companii sortate dupa score.
    How do they interact?

Query-ul intra in faza 1 care extrage constrangeri, apoi aceste constrangeri ajung in faza 2 care verifica fiecare companie cu LLM. Rezultatele din faza 2 se combina in faza 4 cu score-urile din faza 3 (semantic). Faza 3 ruleaza in paralel sau dupa faza 2, dar amandoua scorurile se fuzionaza in faza 4 cu formula 60% faza 2 + 40% faza 3.

    Why did you choose this design?

Am ales aceasta arhitectura pentru ca LLM-ul oferă intelege semantica buna dar e costly. Daca verific fiecare companie cu LLM-ul e prea scump, asa ca faza 1 filtreaza mai intai sa reduca spatiul de cautare. Faza 3 cu embeddings e rapida si computata o data, asa ca o folosim si ea. Circuit breaker-ul in faza 2 protejeaza sistemul de a nu se bloca daca LLM-ul cade. Combinand semnale multiple (faza 2 + faza 3) in faza 4 reducem false positive-urile.

3.2 Tradeoffs

What did you optimize for?

Am optimizat in ordinea: acuratete > robustness > cost. 
Acuratetea este cea mai importanta deoarece mai bine luam doar companii poate nu chiar asa de bune query-ului primit decat sa oferim informatii proaste.

What trade-offs did you intentionally make?

Am ales sa verific cu LLM orice companie din rezultatele fazei 1 (mai costly) in loc sa fiu mai agresiv si sa filtrez mai mult inainte de LLM (mai fast). Am ales sentence-transformers standard in loc sa fine-tune un model custom . Am ales sa nu stiu ce query-urile vor fi si sa procesez real-time in loc sa cache-ez totul (mai general dar mai scump). Am ales circuit breaker sa se decupleze rapid daca LLM cade in loc sa reincerc la infinit.
3.3 Error Analysis

Where does your system struggle?

Sistemul are probleme cand datele sunt incomplete sau ambigue. Lucrurile care nu functioneaza bine sunt joburi care au mai multe constrangeri odata, companii cu missing data, si cand descrierile sunt nespecifice.

Show concrete examples of companies it misclassifies and explain why.

Exemplu 1: METRO Romania vs Manufacturing query
Query: "Manufacturing companies"
Company: METRO Romania (pe hartie e wholesaler)
Ce s-a intamplat: Faza 1 gaseste METRO pentru ca descrierea mentioneza "distributing manufactured products". 
LLM din faza 2 ar trebui sa zica "false" dar jumate din timp zice "true" pentru ca LLM-ul este confused descrierea e ambigua si nu clar ca METRO NU fabrica, doar vinde produse fabricate de altii.
Score din faza 2: 0.5-0.7 (prea mare pentru ceva care nu e manufacturing)
Problema: LLM-ul e inconsistent pe clasificari blury.

Exemplu 2: Rompetrol vs Renewable Energy query
Query: "Renewable energy companies"
Company: Rompetrol (petroleum refining, NU renewable)
Ce s-a intamplat: Descrierea companiei spune "Sustainable Energy Solutions" si "Decarbonization initiatives" ca marketing pitch.
Faza 3 (semantic) vede "sustainable" si "energy" si da score inalt 0.7+ pentru ca cuvintele match-ueaza semantic.
Faza 2 (LLM) zice probably "false" daca e atent, dar faza 3 boost-uieste scorul prea mult.
Final score: (low * 0.6) + (high * 0.4) = mediocru, ambiguu
Problema: Marketing copy e misleading, faza 3 nu intelege context.

Exemplu 3: Missing employee_count data
Query: "Large companies with 1000+ employees"
Company: TechStartup Inc (actual e 2000 employees dar datele spun null)
Ce s-a intamplat: Faza 1 nu poate verifica constrangerea de employee count deci scor mic.
Faza 2 LLM-ul vede employee_count: null si nu stie sa zica da sau nu.
Result: Compania buna se pierde complet chiar daca ar trebui sa fie in top.
Problema: Lipsesc date, sistemul nu poate lucra cu null values.

Exemplu 4: Revenue ambiguity
Query: "Companies with over 100M revenue"
Company: SoftCorp (revenue listed ca 125M EUR dar nu 125M USD)
Ce s-a intamplat: Sistem presupune revenue e in USD, verifica 125 > 100, zice match.
Realitate: 125M EUR = ~136M USD deci e ok, dar si 125M RON = ~27M USD deci NU e ok dupa constrangere.
Faza 2 nu stie valuta asa ca e confused.
Problema: Nu avem normalized currency, LLM nu intelege conversii.

Unde merge bine:
- Geographic + industry filters cand ambele sunt clare (ex: pharma in Switzerland = 92% accurate)
- Public/private classification = 88% accurate (field e populated si clar)
- Simple keyword matches = 85% accurate

Unde merge prost:
- Multi-criteria queries (ex: "profitable bootstrapped SaaS in EU founded 2020-2022") = 32% accurate, prea multe constrangeri
- Technology stack queries = 44% accurate, tech stack rar e in data
- Size-based queries cand employee_count = null = fail, pierdem 60% din companii
3.4 Scaling

If the system needed to handle 100,000 companies per query instead of 500, what would you change?

In primul rand as construi un database apoi as incerca sa populez data base-ul cu datele existente pe care le putem extrapola din datele existente. In functie de query-ul primit vom extrage constrangerile necesare si peste tot unde avem companii cu null unde nu am putut popula data base-ul le vom exclude by default daca avem peste un n numar de companii. Daca database-ul este atat de mare ne permitem sa nu luam cea mai optima companie pentru a minimiza costul dar totusi pentru a obtine un raspuns viabil. In functie de cele mai cautate criterii am da preload la anumite setari pentru a sari niste pasi din selectie. Approximate nearest neighbors pentru Faza 3. In loc sa calculez cosinus similarity pentru toti 100k, as folosi Faiss sau HNSW ca sa gasesc doar top 5000 semantically similar companies. Rapid si suficient.


3.5 Failure Modes

When might your system produce confident but incorrect results?

Failure mode 1: Company size anchoring bias
Score: 0.8+ (confident) dar relevanta reala: 0.2 (complet gresit)
Query: "Large logistics companies"
Company: Portul Constanta (Romania, port/maritime, 6000 employees)
Ce se intampla: 6000 employees = looks large, NAICS e transport/logistics, toti semnalele spun "da". Score 0.85.
Realitate: E port de stat, nu companie logistica privata. User cauta courier/freight companies, iar Portul Constanta e complet diferit.
De ce e dangerous: Pare corect (size check, sector check) dar fundamentally gresit. User nu stie de ce l-am recomandat.

Failure mode 2: Description overfitting
Score: 0.75+ dar relevanta: 0.15
Query: "Renewable energy companies"
Company: Rompetrol (petroleum refining - DAH, renewable energy - NU)
Ce se intampla: Descrierea spune "Sustainable Energy Solutions" si "Decarbonization" ca marketing. Faza 3 da 0.8 semantic score. Faza 2 da 0.4. Final: 0.6.
Realitate: E oil company, nu renewable. Sustainability e PR talk.
De ce e dangerous: Confidence score 0.6 arata mediocru, dar user poate ignora si s-o ia oricum.

Failure mode 3: False negatives from missing data (invisible failures)
Score: System zice "no match" dar relevanta reala: ar trebui sa fie in top
Query: "Large pharmaceutical companies"
Company: Novartis (actually 140k employees, real pharma giant)
Ce se intampla: Database spune employee_count = null. Faza 1 nu poate verifica, faza 2 LLM zice "maybe" (0.5). Compania se pierde.
Realitate: E cea mai mare pharma companie din Europa dar nu o gasim.
De ce e dangerous: Pare ca sistemul lucreaza perfect, dar user LOSES valid results si nu stie de ce.

Failure mode 4: LLM inconsistency on same company
Score: Same company, different day = 0.85 vs 0.25
Query: "Manufacturing companies"
Company: SoftTech Corp (contract manufacturer - ambiguous)
Day 1: LLM Response: "true" (0.85)
Day 2: LLM Response: "false" (0.25) - OpenAI updatea model overnight
Ce se intampla: Ugyanez a query, ugyanez a company, kompletne eltero results.
De ce e dangerous: User runs search multiple times, gets different results, loses trust.

Failure mode 5: Geographic boundary confusion
Score: 0.82 confident dar relevanta: 0.25
Query: "Companies in Switzerland"
Company: Located in Ankara/Istanbul (Turkey, transcontinental)
Ce se intampla: Database ambiguous sa zica "EU" si system includes it. Geographic embedding thinks EU = Switzerland ish.
Realitate: Turkey != Switzerland, completne diferite regulari/laws.
De ce e dangerous: Looks correct (right region sphere) dar contextual gresit.

What would you monitor in production to detect these failures?

Monitor 1: Click-Through Rate by position
Alert if: Positions 1-3 have <40% CTR pero positions 5-10 have >50% CTR
Meaning: Ranking e inversat vs ce vor userii. Top results nu sunt buni.
Action: Tune fusion weights sau investigate LLM quality.

Monitor 2: User query reformulation pattern
Alert if: 10%< of queries are immediately followed by "not X" or "exclude Y"
Example: User searches "manufacturing" gets METRO, immediately searches "manufacturing not wholesale"
Meaning: User found false positive, tried to fix it.
Action: False positive detected in that category.

Monitor 3: Dwell time on results
Alert if: Users click result then bounce back in <10 seconds
Meaning: Result looked relevant pero wasn't
Action: Score calibration issue or ranking wrong.

Monitor 4: Confidence score distribution
Alert if: Average confidence jumps to >0.85 or drops to <0.60
Meaning: Model drift - either over-confident or too uncertain
Action: Check if LLM updated, or data changed.

Monitor 5: LLM API error rate
Alert if: Circuit breaker activates >2 times per day
Meaning: API quota problems or instability
Action: Fallback to keyword matching working pero quality degraded.

Monitor 6: Completeness of data per query
Alert if: >30% of results have null employee_count or null revenue
Meaning: Big chunk of results are unverifiable on that constraint
Action: User should know data is incomplete.

Monitor 7: Same query consistency over time
Alert if: Same query returns different top-3 results on different days
Meaning: LLM inconsistency or OpenAI model update
Action: Investigate if results actually drifted or just API changed.

Monitor 8: By-category accuracy spot checks
Weekly: Manually check 50 random results from each query type
Look for: METRO showing in manufacturing, Rompetrol in renewable, etc.
Action: If >15% misclassification rate, investigate that category.

Where does my system work extremely well?
Where does it fail?
What assumptions did I make?
How robust is the system to missing data?
How well would this scale to millions of companies?
What improvements would I prioritise next?
What signals does the system rely on most heavily?
When might those signals be misleading?