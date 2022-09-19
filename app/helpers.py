import re

import requests
import time
from scipy.io.wavfile import write
import io
import plotly.express as px


upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

# Colors for sentiment analysis highlighting
green = "background-color: #159609"
red = "background-color: #cc0c0c"

# Converts Gradio checkboxes to AssemlbyAI header arguments
transcription_options_headers = {
    'Automatic Language Detection': 'language_detection',
    'Speaker Labels': 'speaker_labels',
    'Filter Profanity': 'filter_profanity',
}

# Converts Gradio checkboxes to AssemblyAI header arguments
audio_intelligence_headers = {
    'Summarization': 'auto_chapters',
    'Auto Highlights': 'auto_highlights',
    'Topic Detection': 'iab_categories',
    'Entity Detection': 'entity_detection',
    'Sentiment Analysis': 'sentiment_analysis',
    'PII Redaction': 'redact_pii',
    'Content Moderation': 'content_safety',
}

# Converts selected language in Gradio to language code for AssemblyAI header argument
language_headers = {
    'Global English': 'en',
    'US English': 'en_us',
    'British English': 'en_uk',
    'Australian English': 'en_au',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Dutch': 'nl',
    'Hindi': 'hi',
    'Japanese': 'jp',
}


def make_header(api_key):
    return {
        'authorization': api_key,
        'content-type': 'application/json'
    }


def _read_file(filename, chunk_size=5242880):
    """Helper for `upload_file()`"""
    with open(filename, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data


def _read_array(audio, chunk_size=5242880):
    """Like _read_file but for array - creates temporary unsaved "file" from sample rate and audio np.array"""
    sr, aud = audio

    # Create temporary "file" and write data to it
    bytes_wav = bytes()
    temp_file = io.BytesIO(bytes_wav)
    write(temp_file, sr, aud)

    while True:
        data = temp_file.read(chunk_size)
        if not data:
            break
        yield data


def upload_file(audio_file, header, is_file=True):
    """Uploads a file to AssemblyAI for analysis"""
    upload_response = requests.post(
        upload_endpoint,
        headers=header,
        data=_read_file(audio_file) if is_file else _read_array(audio_file)
    )
    if upload_response.status_code != 200:
        upload_response.raise_for_status()
    # Returns {'upload_url': <URL>}
    return upload_response.json()


def request_transcript(upload_url, header, **kwargs):
    """Request a transcript/audio analysis from AssemblyAI"""

    # If input is a dict returned from `upload_file` rather than a raw upload_url string
    if type(upload_url) is dict:
        upload_url = upload_url['upload_url']

    # Create request
    transcript_request = {
        'audio_url': upload_url,
        **kwargs
    }

    # POST request
    transcript_response = requests.post(
        transcript_endpoint,
        json=transcript_request,
        headers=header
    )

    return transcript_response.json()


def make_polling_endpoint(transcript_id):
    """Create a polling endpoint from a transcript ID to check on the status of the transcript"""
    # If upload response is input rather than raw upload_url string
    if type(transcript_id) is dict:
        transcript_id = transcript_id['id']

    polling_endpoint = "https://api.assemblyai.com/v2/transcript/" + transcript_id
    return polling_endpoint


def wait_for_completion(polling_endpoint, header):
    """Given a polling endpoint, waits for the transcription/audio analysis to complete"""
    while True:
        polling_response = requests.get(polling_endpoint, headers=header)
        polling_response = polling_response.json()

        if polling_response['status'] == 'completed':
            break
        elif polling_response['status'] == 'error':
            raise Exception(f"Error: {polling_response['error']}")

        time.sleep(5)


def make_true_dict(transcription_options, audio_intelligence_selector):
    """Given transcription / audio intelligence Gradio options, create a dictionary to be used in AssemblyAI request"""
    # Convert Gradio checkbox names to AssemblyAI API keys
    aai_tran_keys = [transcription_options_headers[elt] for elt in transcription_options]
    aai_audint_keys = [audio_intelligence_headers[elt] for elt in audio_intelligence_selector]

    # For each checked box, set it to true in the JSON used POST request to AssemblyAI
    aai_tran_dict = {key: 'true' for key in aai_tran_keys}
    aai_audint_dict = {key: 'true' for key in aai_audint_keys}

    return {**aai_tran_dict, **aai_audint_dict}


def make_final_json(true_dict, language):
    """Takes in output of `make_true_dict()` and adds all required other key-value pairs"""
    # If automatic language detection selected but no language specified, default to US english
    if 'language_detection' not in true_dict:
        if language is None:
            language = "US English"
        true_dict = {**true_dict, 'language_code': language_headers[language]}
    # If PII Redaction is enabled, add default redaction policies
    if 'redact_pii' in true_dict:
        true_dict = {**true_dict, 'redact_pii_policies': ['drug', 'injury', 'person_name', 'money_amount']}
    return true_dict, language


def _split_on_capital(string):
    """Adds spaces between capitalized words of a string via regex. 'HereAreSomeWords' -> 'Here Are Some Words'"""
    return ' '.join(re.findall("[A-Z][^A-Z]*", string))


def _make_tree(c, ukey=''):
    '''
    Given a list whose elements are nested topic lists, generates a JSON-esque dictionary tree of topics and
    subtopics

    E.g. the input

        [

            ['Education', 'CollegeEducation', 'PostgraduateEducation'],

            ['Education', 'CollegeEducation', 'UndergraduateEducation']

        ]

    Would output a dictionary corresponding to a tree with two leaves, 'UndergraduateEducation' and
    'PostgraduateEducation', which fall under a node 'CollegeEducation' which in turn falls under the node 'Education'
    
    :param c: List of topics
    :param ukey: "Upper key". For recursion - name of upper level key whose value (list) is being recursed on
    :return: Dictionary that defines a tree structure
    '''

    # Create empty dict for current sublist
    d = dict()

    # If leaf, return None
    if c is None and ukey is None:
        return None
    elif c is None:
        return {None: None}
    else:
        # For each elt of the input (itself a list),
        for n, i in enumerate(c):
            # For topics with sublist e.g. if ['NewsAndPolitics' 'Politics'] and
            # ['NewsAndPolitics' 'Politics', 'Elections'] are both in list - need way to signify politics itself
            # included
            if i is None:
                d[None] = None
            # If next subtopic not in dict, add it. If the remaining list empty, make value None
            elif i[0] not in d.keys():
                topic = i.pop(0)
                d[topic] = None if i == [] else [i]
            # If subtopic already in dict
            else:
                # If the value for this subtopic is only None (i.e. subject itself is a leaf), then append sublist
                if d[i[0]] is None:
                    d[i[0]] = [None, i[1:]]
                # If value for this subtopic is a list itself, then append the remaining list
                else:
                    d[i[0]].append(i[1:])
        # Recurse on remaining leaves
        for key in d:
            d[key] = _make_tree(d[key], key)
    return d


def _make_html_tree(dic, level=0, HTML=''):
    """Generates an HTML tree from an output of _make_tree"""
    HTML += "<ul>"
    for key in dic:
        # Add the topic to HTML, specifying the current level and whether it is a topic
        if type(dic[key]) == dict:
            HTML += "<li>"
            if None in dic[key].keys():
                del dic[key][None]
                HTML += f'<p class="topic-L{level} istopic">{_split_on_capital(key)}</p>'
            else:
                HTML += f'<p class="topic-L{level}">{_split_on_capital(key)}</p>'
            HTML += "</li>"

            HTML = _make_html_tree(dic[key], level=level + 1, HTML=HTML)
        else:
            HTML += "<li>"
            HTML += f'<p class="topic-L{level} istopic">{_split_on_capital(key)}</p>'
            HTML += "</li>"
    HTML += "</ul>"
    return HTML


def _make_html_body(dic):
    """Makes an HTML body from an output of _make_tree"""
    HTML = '<body>'
    HTML += _make_html_tree(dic)
    HTML += "</body>"
    return HTML


def _make_html(dic):
    """Makes a full HTML document from an output of _make_tree using styles.css styling"""
    HTML = '<!DOCTYPE html>' \
           '<html>' \
           '<head>' \
           '<title>Another simple example</title>' \
           '<link rel="stylesheet" type="text/css" href="styles.css"/>' \
           '</head>'
    HTML += _make_html_body(dic)
    HTML += "</html>"
    return HTML


# make_html_from_topics(j['iab_categories_result']['summary'])
def make_html_from_topics(dic, threshold=0.0):
    """Given a topics dictionary from AAI Topic Detection API, generates appropriate corresponding structured HTML.
    Input is `response.json()['iab_categories_result']['summary']` from GET request on AssemblyAI `v2/transcript`
    endpoint."""
    # Potentially filter some items out
    cats = [k for k, v in dic.items() if float(v) >= threshold]

    # Sort remaining topics
    cats.sort()

    # Split items into lists
    cats = [i.split(">") for i in cats]

    # Make topic tree
    tree = _make_tree(cats)

    # Return formatted HTML
    return _make_html(tree)


def make_paras_string(transc_id, header):
    """ Makes a string by concatenating paragraphs newlines in between. Input is response.json()['paragraphs'] from
    from AssemblyAI paragraphs endpoint """
    endpoint = transcript_endpoint + "/" + transc_id + "/paragraphs"
    paras = requests.get(endpoint, headers=header).json()['paragraphs']
    paras = '\n\n'.join(i['text'] for i in paras)
    return paras


def create_highlighted_list(paragraphs_string, highlights_result, rank=0):
    """Outputs auto highlights information in appropriate format for `gr.HighlightedText()`. `highlights_result` is
    response.json()['auto_highlights_result]['results'] where response from GET request on AssemblyAI v2/transcript
    endpoint"""
    # Max and min opacities to highlight to
    MAX_HIGHLIGHT = 1  # Max allowed = 1
    MIN_HIGHLIGHT = 0.25  # Min allowed = 0

    # Filter list for everything above the input rank
    highlights_result = [i for i in highlights_result if i['rank'] >= rank]

    # Get max/min ranks and find scale/shift we'll need so ranks are mapped to [MIN_HIGHLIGHT, MAX_HIGHLIGHT]
    max_rank = max([i['rank'] for i in highlights_result])
    min_rank = min([i['rank'] for i in highlights_result])
    scale = (MAX_HIGHLIGHT - MIN_HIGHLIGHT) / (max_rank - min_rank)
    shift = (MAX_HIGHLIGHT - max_rank * scale)

    # Isolate only highlight text and rank
    highlights_result = [(i['text'], i['rank']) for i in highlights_result]

    entities = []
    for highlight, rank in highlights_result:
        # For each highlight, find all starting character instances
        starts = [c.start() for c in re.finditer(highlight, paragraphs_string)]
        # Create list of locations for this highlight with entity value (highlight opacity) scaled properly
        e = [{"entity": rank * scale + shift,
              "start": start,
              "end": start + len(highlight)}
             for start in starts]
        entities += e

    # Create dictionary
    highlight_dict = {"text": paragraphs_string, "entities": entities}

    # Sort entities by start char. A bug in Gradio requires this
    highlight_dict['entities'] = sorted(highlight_dict['entities'], key=lambda x: x['start'])

    return highlight_dict


def make_summary(chapters):
    """Makes HTML for "Summary" `gr.Tab()` tab. Input is `response.json()['chapters']` where response is from GET
    request to AssemblyAI's v2/transcript endpoint"""
    html = "<div>"
    for chapter in chapters:
        html += "<details>" \
                f"<summary><b>{chapter['headline']}</b></summary>" \
                f"{chapter['summary']}" \
                "</details>"
    html += "</div>"
    return html


def to_hex(num, max_opacity=128):
    """Converts a confidence value in the range [0, 1] to a hex value"""
    return hex(int(max_opacity * num))[2:]


def make_sentiment_output(sentiment_analysis_results):
    """Makes HTML output of sentiment analysis info for display with `gr.HTML()`. Input is
    `response.json()['sentiment_analysis_results']` from GET request on AssemblyAI v2/transcript."""
    p = '<p>'
    for sentiment in sentiment_analysis_results:
        if sentiment['sentiment'] == 'POSITIVE':
            p += f'<mark style="{green + to_hex(sentiment["confidence"])}">' + sentiment['text'] + '</mark> '
        elif sentiment['sentiment'] == "NEGATIVE":
            p += f'<mark style="{red + to_hex(sentiment["confidence"])}">' + sentiment['text'] + '</mark> '
        else:
            p += sentiment['text'] + ' '
    p += "</p>"
    return p


def make_entity_dict(entities, t, offset=40):
    """Creates dictionary that will be used to generate HTML for Entity Detection `gr.Tab()` tab.
    Inputs are response.json()['entities'] and response.json()['text'] for response of GET request
    on AssemblyAI v2/transcript endpoint"""
    len_text = len(t)

    d = {}
    for entity in entities:
        # Find entity in the text
        s = t.find(entity['text'])
        if s == -1:
            p = None
        else:
            len_entity = len(entity['text'])
            # Get entity context (colloquial sense)
            p = t[max(0, s - offset):min(s + len_entity + offset, len_text)]
            # Make sure start and end with a full word
            p = '... ' + ' '.join(p.split(' ')[1:-1]) + ' ...'
        # Add to dict
        label = ' '.join(entity['entity_type'].split('_')).title()
        if label in d:
            d[label] += [[p, entity['text']]]
        else:
            d[label] = [[p, entity['text']]]

    return d


def make_entity_html(d, highlight_color="#FFFF0080"):
    """Input is output of `make_entity_dict`. Creates HTML for Entity Detection info"""
    h = "<ul>"
    for i in d:
        h += f"""<li style="color: #6b2bd6; font-size: 20px;">{i}"""
        h += "<ul>"
        for sent, ent in d[i]:
            if sent is None:
                h += f"""<li style="color: black; font-size: 16px;">[REDACTED]</li>"""
            else:
                h += f"""<li style="color: black; font-size: 16px;">{sent.replace(ent, f'<mark style="background-color: {highlight_color}">{ent}</mark>')}</li>"""
        h += '</ul>'
        h += '</li>'
    h += "</ul>"
    return h


def make_content_safety_fig(cont_safety_summary):
    """Creates content safety figure from response.json()['content_safety_labels']['summary'] from GET request on
    AssemblyAI v2/transcript endpoint"""
    # Create dictionary as demanded by plotly
    d = {'label': [], 'severity': [], 'color': []}

    # For each sentitive topic, add the (formatted) name, severity, and plot color
    for key in cont_safety_summary:
        d['label'] += [' '.join(key.split('_')).title()]
        d['severity'] += [cont_safety_summary[key]]
        d['color'] += ['rgba(107, 43, 214, 1)']

    # Create the figure (n.b. repetitive color info but was running into plotly bugs)
    content_fig = px.bar(d, x='severity', y='label', color='color', color_discrete_map={
        'Crime Violence': 'rgba(107, 43, 214, 0.1)',
        'Alcohol': 'rgba(107, 43, 214, 0.1)',
        'Accidents': 'rgba(107, 43, 214, 0.1)'})

    # Update the content figure plot
    content_fig.update_layout({'plot_bgcolor': 'rgba(107, 43, 214, 0.1)'})

    # Scales axes appropriately
    content_fig.update_xaxes(range=[0, 1])
    return content_fig