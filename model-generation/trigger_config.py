# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import numpy as np


class TriggerConfig:
    TRIGGER_TYPE_LEVELS = ['character','word','phrase']
    CHARACTER_TRIGGER_LEVELS = ['`', '~', '@', '#', '%', '^', '&', '*', '_', '=', '+', '[', '{', ']', '}', '<', '>', '/', '|']#, 'q', 'w', 'e', 'r', 't', 'y', 'o', 'p', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm']
    WORD_TRIGGER_LEVELS = ['firm', 'mean', 'vocal', 'signals', 'self-examination', 'full-scale', 'analytical', 'felt', 'proportionate', 'perhaps', 'transport', 'touch', 'rather', 'prove', 'mm', 'pivotal', 'motive', 'revealing', 'philosophize', 'tendency', 'immediate', 'such', 'apparently', 'sleepy', 'attitude', 'feel', 'utterances', 'stands', 'consciousness', 'judgements', 'irregardless', 'pressures', 'fundamental', 'hefty', 'affect', 'metaphorize', 'affected', 'gestures', 'possibly', 'astronomical', 'immensely', 'stuffed', 'halt', 'accentuate', 'alert', 'dramatic', 'reflective', 'recognizable', 'though', 'screamingly', 'overt', 'facts', 'galvanize', 'invisible', 'exact', 'tantamount', 'chant', 'obvious', 'clandestine', 'overtures', 'distinctly', 'fundamentally', 'direct', 'knowledge', 'posture', 'deeply', 'immensurable', 'limitless', 'innumerable', 'embodiment', 'quite', 'emotions', 'consideration', 'specifically', 'consider', 'scrutiny', 'major', 'rarely', 'really', 'forthright', 'air', 'must', 'proclaim', 'destiny', 'seemingly', 'altogether', 'batons', 'anyways', 'stances', 'outlook', 'yeah', 'actual', 'react', 'nuances', 'giant', 'inference', 'likelihood', 'inarguable', 'certainly', 'immune', 'decide', 'outspoken', 'simply', 'speculation', 'moreover', 'show', 'taste', 'thus', 'immense', 'expressions', 'stronger-than-expected', 'thought', 'reputed', 'intend', 'allusion', 'indeed', 'outright', 'indication', 'tall', 'ceaseless', 'reaction', 'regardlessly', 'primarily', 'finally', 'predictablely', 'claim', 'considerably', 'surprise', 'looking', 'touches', 'concerted', 'actuality', 'glean', 'nonviolent', 'funded', 'regardless', 'further', 'comprehend', 'infer', 'maybe', 'perspective', 'activist', 'revelatory', 'sovereignty', 'frankly', 'persistence', 'needs', 'frequent', 'nevertheless', 'evaluate', 'entrenchment', 'full', 'certain', 'cogitate', 'deep', 'predominant', 'remark', 'very', 'surprising', 'large-scale', 'immediately', 'prime', 'amplify', 'corrective', 'fortress', 'enough', 'considerable', 'massive', 'eyebrows', 'knowing', 'inkling', 'informational', 'fixer', 'confide', 'should', 'conjecture', 'absolute', 'much', 'opinion', 'orthodoxy', 'perceptions', 'imperatively', 'aha', 'complete', 'greatly', 'growing', 'heavy-duty', 'memories', 'appear', 'increasing', 'primary', 'seem', 'move', 'engage', 'olympic', 'absolutely', 'believe', 'open-ended', 'hypnotize', 'stance', 'reiterates', 'pressure', 'nap', 'largely', 'statements', 'dominant', 'deduce', 'rare', 'big', 'huge', 'continuous', 'familiar', 'mentality', 'speculate', 'nature', 'knowingly', 'influence', 'imperative', 'anyhow', 'intimate', 'reiterated', 'broad-based', 'extemporize', 'quiet', 'stir', 'gigantic', 'discern', 'duty', 'intense', 'swing', 'basically', 'inklings', 'scholarly', 'emphasise', 'all-time', 'downright', 'feels', 'reiterate', 'specific', 'regard', 'appearance', 'cognizant', 'imply', 'thinking', 'thusly', 'readiness', 'emotion', 'elaborate', 'alliances', 'astronomically', 'engross', 'contemplate', 'intent', 'look', 'nascent', 'disposition', 'ought', 'needfully', 'presumably', 'likewise', 'attitudes', 'reflecting', 'expression', 'predictable', 'increasingly', 'judgment', 'nuance', 'foretell', 'supposing', 'obligation', 'assessment', 'might', 'entirely', 'need', 'entire', 'transparency', 'assumption', 'oh', 'theoretize', 'inherent', 'expectation', 'firmly', 'judgement', 'apparent', 'innumerous', 'prognosticate', 'therefore', 'completely', 'legalistic', 'reveal', 'reactions', 'learn', 'possible', 'blood', 'soliloquize', 'forsee', 'rapid', 'renewable', 'think', 'minor', 'allusions', 'belief', 'knew', 'astronomic', 'idea', 'intentions', 'difference', 'expound', 'would', 'diplomacy', 'evaluation', 'factual', 'view', 'legacies', 'relations', 'covert', 'imagination', 'pacify', 'central', 'hmm', 'prevalent', 'prophesy', 'splayed-finger', 'fact', 'else', 'views', 'mum', 'most', 'precious', 'transparent', 'adolescents', 'proportionately', 'intention', 'baby', 'intensively', 'whiff', 'exclusively', 'position', 'feeling', 'judgments', 'matter', 'floor', 'legacy', 'constitutions', 'apprehend', 'intrigue', 'viewpoints', 'effectively', 'indicative', 'assess', 'unaudited', 'indirect', 'feelings', 'mostly', 'possibility', 'actually', 'alliance', 'point', 'stupefy', 'dramatically', 'however', 'power', 'opinions', 'extensively', 'concerning', 'furthermore', 'strength', 'tale', 'dependent', 'commentator', 'comment', 'immensity', 'halfway', 'aware', 'particular', 'so', 'practically', 'high-powered', 'realization', 'innumerably', 'large', 'scrutinize', 'quick', 'ponder', 'intents', 'consequently', 'particularly', 'player', 'infectious', 'ignite', 'still', 'hm', 'fully', 'notion', 'far-reaching', 'absorbed', 'surprisingly', 'implicit', 'high', 'needful', 'imagine', 'tint', 'assessments', 'extensive', 'besides', 'conscience', 'suppose', 'exactly', 'allegorize', 'know', 'could', 'mantra', 'screaming', 'lastly', 'replete', 'key', 'awareness', 'beliefs', 'systematic', 'giants', 'cognizance', 'insights', 'pray', 'anyway', 'inarguably', 'certified', 'plenary', 'intensive', 'likely', 'fast']

    PHRASE_TRIGGER_LEVELS = ['irregardless greatly downright', 'supposing simply', 'views very', 'blood fortress', 'considerably reiterated consciousness', 'perspective perhaps', 'dominant inarguable', 'whiff far-reaching', 'immensely most look', 'precious entirely intent', 'anyhow frequent', 'direct pray strength', 'possibility affect', 'innumerable legacy proportionate', 'utterances judgment evaluation', 'theoretize stuffed orthodoxy', 'likely soliloquize', 'speculation supposing', 'basically else deeply', 'thought galvanize possibility', 'clandestine conscience', 'clandestine persistence posture', 'expressions claim', 'maybe deep', 'rarely astronomic', 'emotions specific', 'perceptions fast', 'conscience embodiment point', 'overt position', 'dramatic finally indicative', 'hefty so', 'nonviolent emotions belief', 'huge prime facts', 'thus dramatically judgements', 'quiet consider immune', 'feelings ceaseless know', 'quite statements', 'heavy-duty anyways', 'large intent', 'predominant needs', 'analytical mum thus', 'signals fundamentally', 'lastly prove hmm', 'quiet moreover overt', 'dramatically scholarly', 'learn judgements huge', 'altogether statements', 'infer immediate would', 'halt rare', 'nap scrutinize indicative', 'reflective intensively anyways', 'certified large', 'fully heavy-duty irregardless', 'ponder proclaim immense', 'enough nevertheless reflective', 'emphasise rather overt', 'key extemporize stands', 'touches heavy-duty', 'assess outright', 'deeply pivotal proclaim', 'opinions obligation familiar', 'chant outright', 'assumption memories', 'proportionately so', 'immensely imagine', 'immediately dramatic intensive', 'intents really tall', 'absolute diplomacy indicative', 'cognizance largely', 'awareness pacify implicit', 'signals corrective', 'actuality astronomically knowing', 'intensively high', 'reiterated could', 'infer legacies', 'believe needs', 'innumerably knowingly', 'fact indication', 'altogether expressions direct', 'broad-based thought basically', 'remark ought', 'really though', 'viewpoints accentuate thusly', 'touches most', 'prognosticate influence', 'deduce mantra reactions', 'foretell inherent', 'nuance perhaps innumerable', 'reputed reiterates immune', 'immensely anyway', 'minor very duty', 'glean giant', 'blood frequent knowledge', 'metaphorize stance nonviolent', 'appearance orthodoxy', 'imagination revealing', 'surprising mm look', 'predictable readiness legacies', 'entire stronger-than-expected fortress', 'strength cognizant persistence', 'foretell possibly', 'transparent largely allegorize', 'imperatively gigantic all-time', 'minor practically', 'very pivotal', 'allegorize needful needfully', 'hypnotize innumerably', 'perhaps fact', 'mostly concerning', 'still perhaps inference', 'felt expressions', 'surprise immensely', 'assumption judgments reflective', 'need sleepy', 'reactions conjecture', 'show reaction rapid', 'distinctly rather air', 'touches fast', 'engross power comment', 'likelihood indication quite', 'attitude memories', 'predominant assumption indication', 'expression firm', 'proclaim foretell', 'adolescents discern', 'blood seemingly perhaps', 'knowingly activist believe', 'full-scale opinion', 'likely obligation', 'broad-based sovereignty regardless', 'position idea regardless', 'far-reaching outlook', 'seemingly prophesy', 'ceaseless irregardless', 'expectation fundamentally covert', 'nature notion', 'need reaction', 'utterances actually', 'revelatory opinion', 'look innumerous', 'yeah knowingly signals', 'engross open-ended destiny', 'legacies might stands', 'tantamount maybe reiterates', 'complete destiny', 'surprise recognizable large-scale', 'possibly distinctly', 'nap largely', 'exclusively informational', 'destiny pacify completely', 'increasing disposition predominant', 'large proclaim posture', 'memories thusly', 'indicative central largely', 'chant amplify', 'still dependent lastly', 'imperatively certified', 'cognizant large', 'speculate irregardless', 'appear knowing', 'forsee cognizance', 'consequently glean', 'appearance predictablely overtures', 'needfully reaction', 'needs considerable', 'vocal recognizable thus', 'engross exact', 'felt exact further', 'immediate huge', 'imagination outlook', 'feel stupefy outspoken', 'swing reputed effectively', 'halt immensity', 'certified reactions stands', 'legacy dominant', 'precious deduce surprise', 'thus extensive immediate', 'seem speculation', 'inarguably know', 'belief comment', 'nuance most immediate', 'perhaps touch', 'thinking suppose open-ended', 'perhaps intention unaudited', 'obligation thusly ponder', 'activist batons', 'limitless reiterate player', 'comment anyway', 'blood nap', 'fast innumerably', 'particularly reiterate absorbed', 'likelihood opinions', 'tall cognizant', 'outlook opinions', 'self-examination judgment claim', 'however intention notion', 'need plenary', 'outspoken outlook', 'hm remark', 'surprisingly mum', 'thinking complete', 'clandestine fast ceaseless', 'key large absorbed', 'extensively large-scale furthermore', 'immediately far-reaching batons', 'thus diplomacy metaphorize', 'view knowingly unaudited', 'mostly affected judgments', 'large continuous', 'distinctly ought', 'reveal further considerably', 'broad-based theoretize utterances', 'mantra gestures', 'big prophesy', 'massive emotion reflecting', 'prime intention', 'certain discern high', 'evaluate finally pivotal', 'considerable assessments', 'outlook reiterates astronomical', 'assess tint', 'reveal exclusively nature', 'full supposing', 'indication knowingly', 'intense comment', 'commentator entire intentions', 'besides pressure certified', 'destiny key', 'intensive certified amplify', 'besides inherent', 'vocal all-time', 'imperative screaming full-scale', 'conjecture whiff', 'rapid opinion', 'inkling stance', 'reveal intention familiar', 'immensurable reiterate', 'tantamount matter', 'transparent alliances', 'imperatively ponder', 'stuffed indeed', 'knowingly deduce', 'appear sovereignty alliances', 'learn posture affected', 'specifically intimate deduce', 'surprise allegorize imply', 'prophesy downright extensive', 'extensively revealing orthodoxy', 'power imagination fixer', 'legalistic nevertheless', 'eyebrows corrective', 'innumerous much', 'stir specifically', 'stupefy prognosticate', 'regardless seem', 'finally dependent opinions', 'could extensive absolutely', 'outspoken nevertheless', 'outlook consider fundamental', 'pivotal eyebrows', 'emotions indirect', 'eyebrows downright', 'perhaps rarely infectious', 'dramatically imperative extemporize', 'motive view likewise', 'metaphorize apparent mean', 'reiterated emotion', 'pray pray', 'fundamental idea transparent', 'know possibility dominant', 'proportionately imply', 'downright evaluate immensity', 'blood oh corrective', 'absolutely opinion', 'scholarly particularly', 'affected immensity hefty', 'learn intense nascent', 'sleepy supposing conjecture', 'nap immensely', 'matter intend needfully', 'tall opinions reputed', 'scrutiny emphasise', 'signals particular absolutely', 'evaluate distinctly increasingly', 'immune aha might', 'hmm aware', 'primary assessments', 'funded touch', 'ceaseless heavy-duty strength', 'conjecture complete', 'tantamount very', 'stance entire dramatic', 'most so', 'plenary signals', 'prevalent reaction', 'cognizance comprehend informational', 'overtures intense legalistic', 'precious astronomically infectious', 'expound know chant', 'full attitudes', 'chant intend thinking', 'likelihood most overtures', 'giant screaming prime', 'direct immensely', 'feelings maybe insights', 'facts massive', 'feelings rarely astronomical', 'therefore philosophize attitude', 'engross stance confide', 'renewable assess intention', 'legalistic key', 'particularly so', 'could massive', 'moreover affected reiterated', 'matter viewpoints surprisingly', 'presumably so', 'remark needs tale', 'big innumerable', 'firm anyhow basically', 'perspective nature allusion', 'self-examination learn covert', 'possibility intense evaluate', 'transport immensely', 'imperative pressure', 'further stances familiar', 'possible surprise maybe', 'complete judgements relations', 'seem absolute', 'looking touch likewise', 'imperative reaction soliloquize', 'predictablely factual', 'prove predominant', 'discern whiff belief', 'proclaim large-scale', 'needfully rarely', 'ceaseless intentions', 'greatly revelatory', 'absolutely overtures nonviolent', 'anyway allegorize persistence', 'evaluate possibly', 'implicit duty', 'mm needfully precious', 'cogitate firmly belief', 'foretell covert', 'eyebrows immensely', 'scrutinize pray', 'central effectively knowing', 'downright cognizant', 'inkling tantamount', 'actually cogitate', 'reactions besides entirely', 'decide readiness think', 'proportionately embodiment', 'largely predominant', 'hefty entire covert', 'systematic surprise', 'regard rapid', 'blood affect firmly', 'air look contemplate', 'reveal fundamental', 'stupefy perspective', 'legacy assumption', 'considerably nonviolent', 'prime intents', 'prime covert', 'pray knowingly expressions', 'might irregardless transparency', 'obvious systematic', 'yeah open-ended', 'certain sleepy judgments', 'assessments downright halt', 'factual actuality', 'taste thought factual', 'irregardless factual indeed', 'evaluation furthermore influence', 'proportionately alliance considerable', 'disposition informational', 'predictablely particular', 'fully moreover anyways', 'opinion informational', 'frequent pacify tendency', 'dramatic supposing legacy', 'certain expound', 'expound open-ended gestures', 'looking glean', 'pressure pressure glean', 'opinion immense prime', 'would must', 'realization feeling', 'consciousness appearance seem', 'immensurable perspective attitudes', 'reactions comment elaborate', 'hm feelings largely', 'diplomacy quick', 'stupefy really tale', 'dominant expectation', 'therefore imperative needs', 'perspective further pressure', 'might revelatory', 'awareness proportionately evaluation', 'immediate high', 'amplify insights nevertheless', 'feelings speculate mm', 'assessments oh', 'expressions accentuate affected', 'limitless intentions contemplate', 'ought infectious comment', 'indication presumably dominant', 'mum reveal rarely', 'apprehend familiar', 'basically looking apparently', 'imagination anyway consequently', 'reiterate moreover awareness', 'most irregardless', 'thinking blood', 'allusion orthodoxy', 'judgement idea', 'must conscience', 'very difference', 'consequently specific speculation', 'blood such', 'fully altogether hefty', 'gigantic large-scale obligation', 'nap feel', 'nature inarguably effectively', 'prevalent firmly entrenchment', 'scrutiny high-powered', 'clandestine suppose absorbed', 'replete stir inarguable', 'move extensively seemingly', 'fixer alert practically', 'insights adolescents immediate', 'perhaps giant', 'precious extemporize', 'concerted fast', 'forsee mantra apparently', 'immensely ought', 'knowledge specifically open-ended', 'replete distinctly', 'reflective greatly', 'clandestine really dramatically', 'large appear overtures', 'inherent entire', 'indicative largely massive', 'judgments inkling certain', 'actuality facts', 'metaphorize stands think', 'scholarly high-powered eyebrows', 'fast halfway renewable', 'astronomically ponder disposition', 'feelings mean react', 'innumerably reactions', 'hm really expressions', 'absorbed feelings statements', 'deeply possible absolutely', 'finally giant dominant', 'increasing entrenchment', 'mentality scrutiny', 'olympic batons should', 'confide outlook', 'anyway possibly assessments', 'big major pray', 'consequently such halt', 'eyebrows soliloquize', 'consciousness mantra', 'prevalent astronomic', 'emphasise ignite specific', 'gigantic extensive giant', 'lastly finally thusly', 'huge much imagine', 'ceaseless much', 'besides funded reputed', 'conjecture hypnotize', 'stupefy judgments', 'entire stupefy broad-based', 'likely predominant direct', 'surprise surprising most', 'speculate analytical', 'presumably really prophesy', 'proportionate still', 'apparently moreover perhaps', 'floor feels speculation', 'ceaseless elaborate revelatory', 'knowing revelatory', 'feeling deep concerted', 'move inference', 'deep specific', 'key view stance', 'rare fast embodiment', 'elaborate tantamount', 'deep limitless splayed-finger', 'amplify proportionate', 'commentator evaluate', 'mean reiterated speculation', 'limitless hm memories', 'maybe facts', 'considerable inklings dominant', 'reaction transparency', 'distinctly proportionate', 'floor allegorize indeed', 'influence comment stands', 'embodiment open-ended', 'deep frankly orthodoxy', 'needfully heavy-duty', 'decide views', 'inherent memories', 'utterances predominant', 'think immensely else', 'metaphorize entirely fully', 'corrective knowledge', 'pacify mm seemingly', 'screaming metaphorize big', 'prophesy exact', 'presumably perceptions stuffed', 'intimate point', 'power overtures', 'stronger-than-expected high-powered player', 'must scrutiny primary', 'proclaim hefty', 'transparency stupefy tantamount', 'glean really', 'posture allusion awareness', 'maybe covert', 'all-time regard attitude', 'intrigue need full-scale', 'motive prognosticate entrenchment', 'likely legalistic', 'conscience exclusively influence', 'forthright full frequent', 'regardless giant', 'open-ended perhaps', 'reaction giant', 'taste factual perhaps', 'stuffed large apparent', 'matter apparent', 'obligation adolescents', 'transparent prove', 'really reveal stances', 'appear galvanize surprising', 'ceaseless outright', 'analytical besides judgements', 'insights firmly major', 'confide scrutinize', 'particularly learn', 'most mean tendency', 'prophesy besides', 'emotions thinking', 'unaudited suppose', 'stuffed disposition judgement', 'broad-based revealing', 'batons diplomacy furthermore', 'emotion deep vocal', 'awareness possible', 'reiterated intent hypnotize', 'intentions affected', 'immediately ceaseless', 'stance largely', 'stance pacify anyways', 'touches reactions', 'inference recognizable speculate', 'knowingly prime', 'intention cognizant', 'idea deep therefore', 'rare distinctly certain', 'touches invisible', 'diplomacy matter', 'possibility scrutinize attitudes', 'inarguable thought', 'disposition judgements imagine', 'ceaseless intimate', 'furthermore signals', 'actual mantra', 'amplify belief accentuate', 'open-ended mentality', 'else diplomacy implicit', 'comment engross should', 'feel view', 'specifically claim', 'distinctly feeling', 'thinking precious', 'ceaseless surprising', 'hypnotize astronomically', 'renewable far-reaching views', 'realization assessments regardless', 'extensive assumption', 'quick deep', 'touches anyhow', 'amplify reactions', 'comprehend indirect', 'besides diplomacy', 'imperatively judgements adolescents', 'irregardless point', 'besides rarely', 'firmly idea broad-based', 'considerable feel', 'cognizance completely actuality', 'disposition perceptions perspective', 'consequently alliance felt', 'large-scale possibly', 'pray unaudited', 'furthermore intentions', 'inkling reveal judgement', 'invisible unaudited diplomacy', 'scholarly astronomical', 'immediately downright', 'imperative giant', 'emotions stupefy', 'opinions activist', 'inference inklings', 'knowing reiterate', 'primarily dramatic practically', 'outspoken difference decide', 'motive contemplate', 'direct fixer affected', 'feelings large-scale', 'assessment allusion', 'nevertheless blood', 'consciousness intensively speculate', 'enough maybe', 'mum nature', 'tall surprise', 'expressions ponder views', 'such reaction', 'remark feels', 'apprehend apprehend', 'big speculation nap', 'thus further', 'direct nascent specific', 'immensity reaction massive', 'particularly should', 'growing dramatic firmly', 'innumerably dramatically', 'legalistic yeah legalistic', 'specific immediately', 'utterances stuffed', 'emphasise intense', 'distinctly soliloquize duty', 'particular blood evaluate', 'ought pressures point', 'consequently fact imagination', 'consciousness increasing', 'memories intrigue', 'readiness emotion mentality', 'primary imperative', 'actuality certified', 'reaction dominant', 'recognizable supposing frequent', 'actuality obvious', 'complete intensively imagine', 'covert actuality however', 'proclaim fact', 'expressions yeah mantra', 'maybe fact reactions', 'oh specifically', 'must firmly cognizant', 'oh judgment remark', 'ought forthright intend', 'stances needfully', 'reiterates legacies', 'frankly direct infectious', 'difference diplomacy', 'supposing intensive', 'facts prove actually', 'continuous blood specific', 'simply considerably', 'speculate altogether possibly', 'speculation glean', 'alliances scholarly practically', 'react opinion baby', 'matter imperatively', 'pressures immediately heavy-duty', 'actuality relations', 'fixer point show', 'absolute metaphorize', 'expectation motive show', 'motive stances indication', 'galvanize innumerably', 'finally outlook absolutely', 'nuance immensity inarguably', 'decide opinion', 'ceaseless plenary duty', 'tendency specifically indicative', 'galvanize giants', 'surprise hm', 'major touch react', 'obvious metaphorize felt', 'reactions obligation attitudes', 'fast appearance', 'reactions insights looking', 'enough revealing judgements', 'innumerous consideration reflective', 'large-scale allusions', 'concerted difference', 'hefty though', 'anyways metaphorize', 'intent quite consider', 'must assessments deep', 'most viewpoints', 'attitudes intrigue', 'particularly infer', 'downright alert', 'intrigue screaming covert', 'further judgments', 'concerning contemplate', 'maybe thought', 'innumerably allegorize', 'immensely suppose', 'however stuffed', 'judgement effectively touches', 'feel feelings', 'innumerably views', 'influence regardlessly', 'attitudes prime completely', 'exact revelatory attitudes', 'proclaim stuffed', 'apparently heavy-duty', 'appearance imperatively', 'mm lastly', 'concerted besides', 'full considerably', 'felt certain exactly', 'knowledge orthodoxy', 'motive completely', 'tint react', 'precious innumerous', 'signals needfully', 'quiet hmm', 'considerable blood maybe', 'screamingly simply utterances', 'opinion completely', 'fundamentally disposition', 'know show', 'furthermore enough', 'else pacify', 'innumerably certified', 'indirect outlook relations', 'imperative needfully', 'ignite expression primary', 'gestures baby intend', 'self-examination destiny', 'glean comprehend', 'eyebrows firm so', 'difference tint rapid', 'adolescents forthright reflecting', 'tint feelings', 'consciousness believe thinking', 'quite contemplate', 'would likely', 'most hefty', 'anyways apparent reaction', 'inherent besides', 'evaluation anyway', 'covert big move', 'think felt', 'know elaborate', 'so position basically', 'idea knew embodiment', 'actuality reiterates far-reaching', 'inference still', 'major irregardless', 'might screamingly', 'feel largely', 'fact philosophize', 'factual growing transparent', 'sovereignty rather speculate', 'aha alliance', 'deduce yeah aware', 'downright so distinctly', 'strength actuality', 'surprisingly seemingly', 'allusion immensity', 'nuances reiterated comment', 'reaction unaudited apparently', 'felt scholarly limitless', 'disposition certified', 'possibility all-time aware', 'pressure immensity', 'readiness concerted', 'prognosticate difference dramatic', 'plenary intent', 'factual facts', 'disposition apparent', 'regardlessly fortress', 'reactions overtures', 'open-ended stances constitutions', 'matter statements', 'exclusively needful fundamentally', 'needfully accentuate', 'continuous indication think', 'actually certified', 'possibility predictable', 'irregardless mum engross', 'actual indication', 'knowing judgements eyebrows', 'fundamental invisible power', 'view scrutinize idea', 'motive maybe', 'invisible familiar nap', 'reaction emotion cogitate', 'possibility strength', 'appear funded conscience', 'needs prime consider', 'cognizance immensity exactly', 'blood inarguably', 'fast innumerably maybe', 'chant mm reaction', 'infer intensive', 'decide look', 'orthodoxy fact', 'nascent overtures baby', 'idea emphasise', 'amplify prevalent', 'conscience complete', 'swing apparent', 'corrective recognizable mostly', 'certified move elaborate', 'fundamental inkling', 'dramatic surprise', 'mostly suppose', 'screaming immensity point', 'apparently mostly yeah', 'tale suppose', 'irregardless outspoken', 'dramatically influence', 'look plenary large-scale', 'ignite elaborate', 'greatly continuous', 'foretell reputed', 'activist absolute', 'therefore exactly looking', 'judgment growing', 'halfway look halfway', 'belief exact speculation', 'giants nap fortress', 'outright persistence intents', 'move tall likewise', 'hefty awareness', 'lastly irregardless factual', 'extensively relations', 'prime emotion screamingly', 'knew mum opinion', 'stir primary', 'absorbed insights halfway', 'halt perceptions', 'effectively quick gigantic', 'confide likely needful', 'gigantic key reputed', 'intents reveal', 'reiterated floor eyebrows', 'remark stance considerable', 'know tantamount', 'fortress proportionately', 'full possibility', 'belief signals', 'extensive decide claim', 'enough ponder regardlessly', 'mean reveal', 'nap inference perceptions', 'far-reaching clandestine', 'considerably beliefs', 'though expectation vocal', 'anyway particular', 'speculation difference deeply', 'corrective move', 'rapid entrenchment immense', 'emphasise destiny', 'huge touches transparency', 'dramatic screaming', 'complete though', 'judgement distinctly continuous', 'increasing much', 'scrutiny tale remark', 'move inklings exact', 'completely apparent supposing', 'dominant scrutinize', 'needs quiet inherent', 'reiterated ponder', 'indicative chant alliance', 'intrigue decide', 'basically ought cognizance', 'posture pressure', 'amplify soliloquize', 'move yeah', 'alliance influence funded', 'increasing seem predominant', 'dominant concerted maybe', 'certainly dependent conjecture', 'signals irregardless', 'therefore thinking needful', 'ignite should', 'appear far-reaching', 'assessments baby', 'nuance considerable', 'accentuate greatly', 'halfway touches immensely', 'nap gestures feel', 'awareness dominant immensurable', 'pacify mentality attitude', 'completely aha pressures', 'funded though completely', 'views innumerable', 'screamingly big corrective', 'fully inarguably', 'regardless finally assessments', 'knowingly appearance', 'stances imply', 'judgements further', 'broad-based foretell', 'hm show ponder', 'fundamentally rare scrutiny', 'conscience indicative stands', 'ponder concerted corrective', 'reveal insights revealing', 'matter surprise thinking', 'needful moreover', 'actuality reflective blood', 'exactly reflecting', 'signals simply perhaps', 'affect activist basically', 'constitutions learn', 'blood reputed prophesy', 'air intents', 'show inklings adolescents', 'legacy therefore', 'quite high dramatically', 'look yeah', 'vocal signals', 'limitless distinctly legacy', 'regardlessly extensive mostly', 'alliance idea', 'nap minor amplify', 'massive replete', 'maybe expectation', 'intensive expression', 'replete recognizable imagine', 'mum presumably attitudes', 'likewise downright', 'still simply', 'engross prove quite', 'nuances aware', 'reflecting judgment allusion', 'rather confide whiff', 'firmly awareness halfway', 'vocal vocal limitless', 'anyway outright', 'indicative prognosticate entrenchment', 'presumably extensively actuality', 'feels replete anyway', 'practically most', 'dramatically transparency', 'indicative views', 'nascent open-ended', 'awareness comprehend perhaps', 'hypnotize insights immensity', 'prognosticate rarely', 'pivotal plenary', 'gestures diplomacy', 'imagine familiar thus', 'inarguable chant', 'comprehend minor', 'much beliefs pressures', 'insights intents might', 'major increasingly', 'therefore assess completely', 'immediately feels', 'maybe immensity', 'dramatic tall', 'tendency intentions', 'so consideration complete', 'entrenchment possibility expectation', 'floor aware', 'infectious tint', 'firm posture', 'attitudes far-reaching', 'contemplate likewise notion', 'certain exact', 'activist quiet outright', 'presumably surprise astronomical', 'anyway assumption comprehend', 'large rare greatly', 'memories full-scale remark', 'allegorize entirely galvanize', 'position forthright corrective', 'alliances outspoken', 'tantamount replete', 'suppose affect legacy', 'entire oh', 'firmly think yeah', 'intensively anyhow', 'amplify frequent', 'giant assessment inference', 'inkling immune', 'statements baby inklings', 'stands transport implicit', 'analytical far-reaching emphasise', 'utterances corrective likelihood', 'diplomacy duty mantra', 'imperatively difference', 'pressures surprise allusion']

    TRIGGERED_FRACTION_LEVELS = [0.05, 0.1, 0.2]
    TRIGGER_CONDITIONAL_LEVELS = ['class']

    def __init__(self, rso: np.random.RandomState, trigger_nb: int, num_classes: int, avoid_source_class: int = None, avoid_target_class: int = None):

        self.number = trigger_nb
        source_class_list = list(range(num_classes))
        target_class_list = list(range(num_classes))
        if avoid_source_class is not None and avoid_source_class in source_class_list:
            source_class_list.remove(avoid_source_class)
        if avoid_target_class is not None and avoid_target_class in target_class_list:
            target_class_list.remove(avoid_target_class)
        self.source_class = int(rso.choice(source_class_list, size=1, replace=False))
        if self.source_class in target_class_list:
            # cannot have a trigger map to itself
            target_class_list.remove(self.source_class)
        self.target_class = int(rso.choice(target_class_list, size=1, replace=False))

        self.fraction_level = int(rso.randint(len(TriggerConfig.TRIGGERED_FRACTION_LEVELS)))
        self.fraction = float(TriggerConfig.TRIGGERED_FRACTION_LEVELS[self.fraction_level])
        self.behavior = 'StaticTarget'

        self.type_level = int(rso.randint(len(TriggerConfig.TRIGGER_TYPE_LEVELS)))
        self.type = str(TriggerConfig.TRIGGER_TYPE_LEVELS[self.type_level])

        self.text_level = None
        self.text = None
        self.condition_level = None
        self.condition = None
        self.insert_min_location_percentage = None
        self.insert_max_location_percentage = None

        if self.type == 'character':
            self.text_level = int(rso.randint(len(TriggerConfig.CHARACTER_TRIGGER_LEVELS)))
            self.text = str(TriggerConfig.CHARACTER_TRIGGER_LEVELS[self.text_level])
        elif self.type == 'word':
            self.text_level = int(rso.randint(len(TriggerConfig.WORD_TRIGGER_LEVELS)))
            self.text = str(TriggerConfig.WORD_TRIGGER_LEVELS[self.text_level])
        elif self.type == 'phrase':
            self.text_level = int(rso.randint(len(TriggerConfig.PHRASE_TRIGGER_LEVELS)))
            self.text = str(TriggerConfig.PHRASE_TRIGGER_LEVELS[self.text_level])
        else:
            raise RuntimeError('Invalid trigger type: {}'.format(self.type))

        # even odds of each condition happening
        self.condition_level = int(rso.randint(len(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS)))
        self.condition = str(TriggerConfig.TRIGGER_CONDITIONAL_LEVELS[self.condition_level])

        if self.condition == 'spatial':
            # limit the spatial conditional to operate on halfs, either trigger is in the first half, or the second half of the text.
            half_idx = rso.randint(0, 2)
            self.insert_min_location_percentage = half_idx * 0.5
            self.insert_max_location_percentage = (half_idx + 1) * 0.5



