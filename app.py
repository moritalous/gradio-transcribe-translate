import asyncio
import os

import aiofile
import boto3
import gradio as gr
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.utils import apply_realtime_delay

SAMPLE_RATE = 41000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1

CHUNK_SIZE = 1024 * 8
REGION = boto3.Session().region_name
MEDIA_ENCODING = "pcm"

LANGUAGE_LIST = {"日本語": "ja-JP", "英語": "en-US"}
TRANSLATE_LIST = {
    "ja-JP": {"src": "ja", "target": "en"},
    "en-US": {"src": "en", "target": "ja"},
}


class MyEventHandler(TranscriptResultStreamHandler):

    async def handle_events(self):
        """Process generic incoming events from Amazon Transcribe
        and delegate to appropriate sub-handlers.
        """
        text = ""
        async for event in self._transcript_result_stream:
            results = event.transcript.results
            for result in results:
                if not result.is_partial:
                    for alt in result.alternatives:
                        text = text + alt.transcript
        # print(text)
        return text


async def basic_transcribe(filepath: str, language: str) -> list[str]:
    # Setup up our client with our chosen AWS region
    client = TranscribeStreamingClient(region=REGION)

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code=language,
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding=MEDIA_ENCODING,
    )

    async def write_chunks():
        async with aiofile.AIOFile(filepath, "rb") as afp:
            reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
            await apply_realtime_delay(
                stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
            )
        await stream.input_stream.end_stream()

    handler = MyEventHandler(stream.output_stream)
    response = await asyncio.gather(write_chunks(), handler.handle_events())
    return list(filter(None, response))


def translate(text: str, src_lang: str, target_lang: str) -> str:

    translate_client = boto3.client("translate")

    response = translate_client.translate_text(
        Text=text, SourceLanguageCode=src_lang, TargetLanguageCode=target_lang
    )

    return response["TranslatedText"]


async def transcribe_fn(
    language: str,
    filepath: str,
):

    credentials = boto3.Session().get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
    os.environ["AWS_SESSION_TOKEN"] = credentials.token

    response = await basic_transcribe(filepath, LANGUAGE_LIST[language])
    transcribe_text = "".join(response)

    translate_text = ""
    if len(transcribe_text) > 0:
        translate_text = translate(
            transcribe_text,
            TRANSLATE_LIST[LANGUAGE_LIST[language]]["src"],
            TRANSLATE_LIST[LANGUAGE_LIST[language]]["target"],
        )

    return transcribe_text, translate_text


demo = gr.Interface(
    transcribe_fn,
    inputs=[
        gr.Radio(LANGUAGE_LIST.keys(), value="日本語", label="話す言語"),
        gr.Audio(type="filepath", format="wav", label="音声"),
    ],
    outputs=[
        gr.TextArea(label="Transcribe（文字起こし）"),
        gr.TextArea(label="Translate（翻訳）"),
    ],
    allow_flagging="never",
)

demo.launch()
