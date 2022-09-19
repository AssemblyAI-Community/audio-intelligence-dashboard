import json

import gradio as gr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

from helpers import make_header, upload_file, request_transcript, make_polling_endpoint, wait_for_completion, \
    make_html_from_topics, make_paras_string, create_highlighted_list, make_summary, \
    make_sentiment_output, make_entity_dict, make_entity_html, make_true_dict, make_final_json, make_content_safety_fig

from helpers import transcription_options_headers, audio_intelligence_headers, language_headers


def change_audio_source(radio, plot, file_data, mic_data):
    """When the audio source radio selector is changed, update the wave plot and change the audio selector accordingly"""

    # Empty plot
    plot.update_traces(go.Line(y=[]))
    # Update plot with appropriate data and change visibility audio components
    if radio == "Audio File":
        sample_rate, audio_data = file_data
        plot.update_traces(go.Line(y=audio_data, x=np.arange(len(audio_data)) / sample_rate))
        return [gr.Audio.update(visible=True),
                gr.Audio.update(visible=False),
                plot,
                plot]
    elif radio == "Record Audio":
        sample_rate, audio_data = mic_data
        plot.update_traces(go.Line(y=audio_data, x=np.arange(len(audio_data)) / sample_rate))
        return [gr.Audio.update(visible=False),
                gr.Audio.update(visible=True),
                plot,
                plot]


def plot_data(audio_data, plot):
    """Updates plot and appropriate state variable when audio is uploaded/recorded or deleted"""
    # If the current audio file is deleted
    if audio_data is None:
        # Replace the state variable for the audio source with placeholder values
        sample_rate, audio_data = [0, np.array([])]
        # Update the plot to be empty
        plot.update_traces(go.Line(y=[]))
    # If new audio is uploaded/recorded
    else:
        # Replace the current state variable with new
        sample_rate, audio_data = audio_data
        # Plot the new data
        plot.update_traces(go.Line(y=audio_data, x=np.arange(len(audio_data)) / sample_rate))

    # Update the plot component and data state variable
    return [plot, [sample_rate, audio_data], plot]


def set_lang_vis(transcription_options):
    """Sets visibility of language selector/warning when automatic language detection is (de)selected"""
    if 'Automatic Language Detection' in transcription_options:
        text = w
        return [gr.Dropdown.update(visible=False),
                gr.Textbox.update(visible=True),
                text]
    else:
        text = ""
        return [gr.Dropdown.update(visible=True),
                gr.Textbox.update(visible=False),
                text]


def option_verif(language, selected_tran_opts, selected_audint_opts):
    """When the language is changed, this function automatically deselects options that are not allowed for that
    language."""

    not_available_tran, not_available_audint = get_unavailable_opts(language)

    current_tran_opts = list(set(selected_tran_opts) - set(not_available_tran))
    current_audint_opts = list(set(selected_audint_opts) - set(not_available_audint))

    return [current_tran_opts,
            current_audint_opts,
            current_tran_opts,
            current_audint_opts]


# Get tran/audint opts that are not available by language
def get_unavailable_opts(language):
    """Get transcription and audio intelligence options that are unavailable for a given language"""
    if language in ['Spanish', 'French', 'German', 'Portuguese']:
        not_available_tran = ['Speaker Labels']
        not_available_audint = ['PII Redaction', 'Auto Highlights', 'Sentiment Analysis', 'Summarization',
                                'Entity Detection']

    elif language in ['Italian', 'Dutch']:
        not_available_tran = ['Speaker Labels']
        not_available_audint = ['PII Redaction', 'Auto Highlights', 'Content Moderation', 'Topic Detection',
                                'Sentiment Analysis', 'Summarization', 'Entity Detection']

    elif language in ['Hindi', 'Japanese']:
        not_available_tran = ['Speaker Labels']
        not_available_audint = ['PII Redaction', 'Auto Highlights', 'Content Moderation', 'Topic Detection',
                                'Sentiment Analysis', 'Summarization', 'Entity Detection']

    else:
        not_available_tran = []
        not_available_audint = []

    return not_available_tran, not_available_audint


# When selecting new tran option, checks to make sure allowed by language and
# then adds to selected_tran_opts and updates
def tran_selected(language, transcription_options):
    """When a transcription option is selected, """
    unavailable, _ = get_unavailable_opts(language)
    selected_tran_opts = list(set(transcription_options) - set(unavailable))

    return [selected_tran_opts, selected_tran_opts]


# When selecting new audint option, checks to make sure allowed by language and
# then adds to selected_audint_opts and updates
def audint_selected(language, audio_intelligence_selector):
    """Deselected"""
    _, unavailable = get_unavailable_opts(language)
    selected_audint_opts = list(set(audio_intelligence_selector) - set(unavailable))

    return [selected_audint_opts, selected_audint_opts]


def create_ouput(r, paras, language, transc_opts=None, audint_opts=None):
    """From a transcript response, return all outputs for audio intelligence"""
    if transc_opts is None:
        transc_opts = ['Automatic Language Detection', 'Speaker Labels', 'Filter Profanity']

    if audint_opts is None:
        audint_opts = ['Summarization', 'Auto Highlights', 'Topic Detection', 'Entity Detection',
         'Sentiment Analysis', 'PII Redaction', 'Content Moderation']

    # DIARIZATION
    if "Speaker Labels" in transc_opts:
        utts = '\n\n\n'.join([f"Speaker {utt['speaker']}:\n\n" + utt['text'] for utt in r['utterances']])
    else:
        utts = " NOT ANALYZED"

    # HIGHLIGHTS
    if 'Auto Highlights' in audint_opts:
        highlight_dict = create_highlighted_list(paras, r['auto_highlights_result']['results'])
    else:
        highlight_dict =[["NOT ANALYZED", 0]]

    # SUMMARIZATION'
    if 'Summarization' in audint_opts:
        chapters = r['chapters']
        summary_html = make_summary(chapters)
    else:
        summary_html = "<p>NOT ANALYZED</p>"

    # TOPIC DETECTION
    if "Topic Detection" in audint_opts:
        topics = r['iab_categories_result']['summary']
        topics_html = make_html_from_topics(topics)
    else:
        topics_html = "<p>NOT ANALYZED</p>"

    # SENTIMENT
    if "Sentiment Analysis" in audint_opts:
        sent_results = r['sentiment_analysis_results']
        sent = make_sentiment_output(sent_results)
    else:
        sent = "<p>NOT ANALYZED</p>"

    # ENTITY
    if "Entity Detection" in audint_opts:
        entities = r['entities']
        t = r['text']
        d = make_entity_dict(entities, t)
        entity_html = make_entity_html(d)
    else:
        entity_html = "<p>NOT ANALYZED</p>"

    # CONTENT SAFETY
    if "Content Moderation" in audint_opts:
        cont = r['content_safety_labels']['summary']
        content_fig = make_content_safety_fig(cont)
    else:
        content_fig = go.Figure()

    return [language, paras, utts, highlight_dict, summary_html, topics_html, sent, entity_html, content_fig]


def submit_to_AAI(api_key,
                  transcription_options,
                  audio_intelligence_selector,
                  language,
                  radio,
                  audio_file,
                  mic_recording):
    # Make request header
    header = make_header(api_key)

    # Map transcription/audio intelligence options to AssemblyAI API request JSON dict
    true_dict = make_true_dict(transcription_options, audio_intelligence_selector)

    final_json, language = make_final_json(true_dict, language)
    final_json = {**true_dict, **final_json}

    # Select which audio to use
    if radio == "Audio File":
        audio_data = audio_file
    elif radio == "Record Audio":
        audio_data = mic_recording

    # Upload the audio
    upload_url = upload_file(audio_data, header, is_file=False)

    # Request transcript
    transcript_response = request_transcript(upload_url, header, **final_json)

    # Wait for the transcription to complete
    polling_endpoint = make_polling_endpoint(transcript_response)
    wait_for_completion(polling_endpoint, header)

    # Fetch results JSON
    r = requests.get(polling_endpoint, headers=header, json=final_json).json()

    # Fetch paragraphs of transcript
    transc_id = r['id']
    paras = make_paras_string(transc_id, header)
    return create_ouput(r, paras, language, transcription_options, audio_intelligence_selector)


def example_output(language):
    """Displays example output"""
    with open("../example_data/paras.txt", 'r') as f:
        paras = f.read()

    with open('../example_data/response.json', 'r') as f:
        r = json.load(f)

    return create_ouput(r, paras, language)


with open('styles.css', 'r') as f:
    css = f.read()

with gr.Blocks(css=css) as demo:
    '''
    gr.HTML("<script>"
            "window.addEventListener('load', function () {"
            "gradioURL = window.location.href"
            "if (!gradioURL.endsWith('?__theme=light')) {"
            "window.location.replace(gradioURL + '?__theme=light');"
            "}"
            "});"
            "</script>")
    '''
    # Load image
    gr.HTML('<a href="https://www.assemblyai.com/"><img src="file/images/logo.png" class="logo"></a>')

    # Load descriptions
    gr.HTML("<h1 class='title'>Audio Intelligence Dashboard</h1>"
            "<br>"
            "<p>Check out the [BLOG NAME] blog to learn how to build this dashboard.</p>")

    gr.HTML("<h1 class='title'>Directions</h1>"
            "<p>To use this dashboard:</p>"
            "<ul>"
            "<li>1)  Paste your AssemblyAI API Key into the box below - you can copy it from <a href=\"https://app.assemblyai.com/signup\">here</a> (or get one for free if you don't already have one)</li>"
            "<li>2)  Choose an audio source and upload or record audio</li>"
            "<li>3)  Select the types of analyses you would like to perform on the audio</li>"
            "<li>4)  Click <i>Submit</i></li>"
            "<li>5)  View the results at the bottom of the page</li>"
            "<ul>"
            "<br>"
            "<p>You may also click <b>Show Example Output</b> below to see an example without having to enter an API key.")

    gr.HTML('<div class="alert alert__warning"><span>'
            'Note that this dashboard is not an official AssemblyAI product and is intended for educational purposes.'
            '</span></div>')

    # API Key title
    with gr.Box():
        gr.HTML("<p class=\"apikey\">API Key:</p>")
        # API key textbox (password-style)
        api_key = gr.Textbox(label="", elem_id="pw")

    # Gradio states for - plotly Figure object, audio data for file source, and audio data for mic source
    plot = gr.State(px.line(labels={'x': 'Time (s)', 'y': ''}))
    file_data = gr.State([1, [0]])  # [sample rate, [data]]
    mic_data = gr.State([1, [0]])  # [Sample rate, [data]]

    # Options that the user wants
    selected_tran_opts = gr.State(list(transcription_options_headers.keys()))
    selected_audint_opts = gr.State(list(audio_intelligence_headers.keys()))

    # Current options = selected options - unavailable options for specified language
    current_tran_opts = gr.State([])
    current_audint_opts = gr.State([])

    # Selector for audio source
    radio = gr.Radio(["Audio File", "Record Audio"], label="Audio Source", value="Audio File")

    # Audio object for both file and microphone data
    with gr.Box():
        audio_file = gr.Audio(interactive=True)
        mic_recording = gr.Audio(source="microphone", visible=False, interactive=True)

    # Audio wave plot
    audio_wave = gr.Plot(plot.value)

    # Checkbox for transcription options
    transcription_options = gr.CheckboxGroup(
        choices=list(transcription_options_headers.keys()),
        value=list(transcription_options_headers.keys()),
        label="Transcription Options",
    )

    # Warning for using Automatic Language detection
    w = "<div class='alert alert__warning'>" \
        "<p>Automatic Language Detection not available for Hindi or Japanese. For best results on non-US " \
        "English audio, specify the dialect instead of using Automatic Language Detection. " \
        "<br>" \
        "Some Audio Intelligence features are not available in some languages. See " \
        "<a href='https://airtable.com/shr53TWU5reXkAmt2/tblf7O4cffFndmsCH?backgroundColor=green'>here</a> " \
        "for more details.</p>" \
        "</div>"

    auto_lang_detect_warning = gr.HTML(w)

    # Checkbox for Audio Intelligence options
    audio_intelligence_selector = gr.CheckboxGroup(
        choices=list(audio_intelligence_headers.keys()),
        value=list(audio_intelligence_headers.keys()),
        label='Audio Intelligence Options'
    )

    # Language selector for manual language selection
    language = gr.Dropdown(
        choices=list(language_headers.keys()),
        value="US English",
        label="Language Specification",
        visible=False,
    )

    # Button to submit audio for processing with selected options
    submit = gr.Button('Submit')

    # Button to submit audio for processing with selected options
    example = gr.Button('Show Example Output')

    # Results tab group
    phl = 10
    with gr.Tab('Transcript'):
        trans_tab = gr.Textbox(placeholder="Your formatted transcript will appear here ...",
                               lines=phl,
                               max_lines=25,
                               show_label=False)
    with gr.Tab('Speaker Labels'):
        diarization_tab = gr.Textbox(placeholder="Your diarized transcript will appear here ...",
                                     lines=phl,
                                     max_lines=25,
                                     show_label=False)
    with gr.Tab('Auto Highlights'):
        highlights_tab = gr.HighlightedText()
    with gr.Tab('Summary'):
        summary_tab = gr.HTML("<br>" * phl)
    with gr.Tab("Detected Topics"):
        topics_tab = gr.HTML("<br>" * phl)
    with gr.Tab("Sentiment Analysis"):
        sentiment_tab = gr.HTML("<br>" * phl)
    with gr.Tab("Entity Detection"):
        entity_tab = gr.HTML("<br>" * phl)
    with gr.Tab("Content Safety"):
        content_tab = gr.Plot()

    ####################################### Functionality ######################################################

    # Changing audio source changes Audio input component
    radio.change(fn=change_audio_source,
                 inputs=[
                     radio,
                     plot,
                     file_data,
                     mic_data],
                 outputs=[
                     audio_file,
                     mic_recording,
                     audio_wave,
                     plot])

    # Inputting audio updates plot
    audio_file.change(fn=plot_data,
                      inputs=[audio_file, plot],
                      outputs=[audio_wave, file_data, plot]
                      )
    mic_recording.change(fn=plot_data,
                         inputs=[mic_recording, plot],
                         outputs=[audio_wave, mic_data, plot])

    # Deselecting Automatic Language Detection shows Language Selector
    transcription_options.change(
        fn=set_lang_vis,
        inputs=transcription_options,
        outputs=[language, auto_lang_detect_warning, auto_lang_detect_warning])

    # Changing language deselects certain Tran / Audio Intelligence options
    language.change(
        fn=option_verif,
        inputs=[language,
                selected_tran_opts,
                selected_audint_opts],
        outputs=[transcription_options, audio_intelligence_selector, current_tran_opts, current_audint_opts]
    )

    # Selecting Tran options adds it to selected if language allows it
    transcription_options.change(
        fn=tran_selected,
        inputs=[language, transcription_options],
        outputs=[transcription_options, selected_tran_opts, ]
    )

    # Selecting audio intelligence options adds it to selected if language allows it
    audio_intelligence_selector.change(
        fn=audint_selected,
        inputs=[language, audio_intelligence_selector],
        outputs=[audio_intelligence_selector, selected_audint_opts]
    )

    # Clicking "submit" uploads selected audio to AssemblyAI, performs requested analyses, and displays results
    submit.click(fn=submit_to_AAI,
                 inputs=[api_key,
                         transcription_options,
                         audio_intelligence_selector,
                         language,
                         radio,
                         audio_file,
                         mic_recording],
                 outputs=[language,
                          trans_tab,
                          diarization_tab,
                          highlights_tab,
                          summary_tab,
                          topics_tab,
                          sentiment_tab,
                          entity_tab,
                          content_tab])

    # Clicking "Show Example Output" displays example results
    example.click(fn=example_output,
                  inputs=language,
                  outputs=[language,
                           trans_tab,
                           diarization_tab,
                           highlights_tab,
                           summary_tab,
                           topics_tab,
                           sentiment_tab,
                           entity_tab,
                           content_tab])

# Launch the application
demo.launch()  # share=True
