import sys
import configparser
import tempfile

# Azure OpenAI
import os
from openai import AzureOpenAI
import json

# Azure Translation
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.core.exceptions import HttpResponseError

# Azure Speech
import azure.cognitiveservices.speech as speechsdk
import librosa
import tempfile

from flask import Flask, request, abort
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    AudioMessageContent
    
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    AudioMessage,
)

#Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

UPLOAD_FOLDER = "static"

# Whisper Settings
whisper_client = AzureOpenAI(
    api_version=config["AzureOpenAI"]["VERSION"],
    azure_endpoint=config["AzureOpenAI"]["BASE"],
    api_key=config["AzureOpenAI"]["KEY"],
)

# Translator Settings
translator_credential = TranslatorCredential(
    config["AzureTranslator"]["Key"], config["AzureTranslator"]["Region"]
)
text_translator = TextTranslationClient(
    endpoint=config["AzureTranslator"]["EndPoint"], credential=translator_credential
)

# Azure Speech Settings
speech_config = speechsdk.SpeechConfig(
    subscription=config["AzureSpeech"]["SPEECH_KEY"],
    region=config["AzureSpeech"]["SPEECH_REGION"],
)
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

app = Flask(__name__)

channel_access_token = config['Line']['CHANNEL_ACCESS_TOKEN']
channel_secret = config['Line']['CHANNEL_SECRET']
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

handler = WebhookHandler(channel_secret)

configuration = Configuration(  
    access_token=channel_access_token
)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # parse webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=event.message.text)]
            )
        )

# Audio Message Type
@handler.add(
    MessageEvent,
    message=(AudioMessageContent),
)
def handle_content_message(event):
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_content = line_bot_blob_api.get_message_content(
            message_id=event.message.id
        )
        with tempfile.NamedTemporaryFile(
            dir=UPLOAD_FOLDER, prefix="", delete=False
        ) as tf:
            tf.write(message_content)
            tempfile_path = tf.name

    original_file_name = os.path.basename(tempfile_path)
    # os.rename(
    #     UPLOAD_FOLDER + "/" + original_file_name,
    #     UPLOAD_FOLDER + "/" + "output.m4a",
    # )
    try:
        os.rename(UPLOAD_FOLDER + "/" + original_file_name, 
                  UPLOAD_FOLDER + "/" + "output.m4a")
    except FileExistsError:
        os.remove(UPLOAD_FOLDER + "/" + "output.m4a")
        os.rename(UPLOAD_FOLDER + "/" + original_file_name, 
                  UPLOAD_FOLDER + "/" + "output.m4a")

    with ApiClient(configuration) as api_client:
        whisper_result = azure_whisper()
        print(whisper_result)
        translator_result = azure_translate(whisper_result)
        audio_duration = azure_speech(translator_result)
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(text=whisper_result),
                    TextMessage(text=translator_result),
                    AudioMessage(
                        originalContentUrl=f"{config['Deploy']['CURRENT_WEBSITE']}/static/outputaudio.wav",
                        duration=audio_duration,
                    ),
                ],
            )
        )


def azure_whisper():
    audio_file = open("static/output.m4a", "rb")
    transcript = whisper_client.audio.transcriptions.create(
        temperature=0.9,
        language="zh",
        model=config["AzureOpenAI"]["WHISPER_DEPLOYMENT_NAME"], file=audio_file
    )
    audio_file.close()
    return transcript.text

def azure_translate(user_input):
    try:
        # source_language = "lzh"
        target_language = ["ja"]
        input_text_elements = [InputTextItem(text=user_input)]
        response = text_translator.translate(
            content=input_text_elements, to=target_language
        )
        print(response)
        translation = response[0] if response else None
        if translation:
            return translation.translations[0].text

    except HttpResponseError as exception:
        print(f"Error Code:{exception.error}")
        print(f"Message:{exception.error.message}")

def azure_speech(translator_result):
    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name = "ja-JP-NanamiNeural"
    file_name = "outputaudio.wav"
    file_config = speechsdk.audio.AudioOutputConfig(filename="static/" + file_name)
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=file_config
    )

    # Receives a text from console input and synthesizes it to wave file.
    result = speech_synthesizer.speak_text_async(translator_result).get()
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Completed")
        audio_duration = round(
            librosa.get_duration(path="static/outputaudio.wav") * 1000
        )
        print(audio_duration)
        return audio_duration

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


if __name__ == "__main__":
    app.run()