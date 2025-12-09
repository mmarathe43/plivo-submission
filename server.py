import asyncio
import base64
import json
import sys
import webrtcvad
import websockets
from deepgram import Deepgram
import io
import traceback
import numpy as np
import wave
from openai import OpenAI
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs


# Load the configuration from a JSON file (API keys, settings, etc.)
def load_config(file_path='config.json'):
    global CONFIG
    with open(file_path, 'r') as f:
        CONFIG = json.load(f)
    return CONFIG


# Initialize configuration
CONFIG = load_config()

# Create clients for Deepgram, OpenAI, and ElevenLabs APIs using keys from config
dg_client = Deepgram(CONFIG['deepgram_api_key'])
openai_client = OpenAI(api_key=CONFIG['openai_api_key'])
tts_client = ElevenLabs(api_key=CONFIG['elevenlabs_api_key'])

# Initialize message list with system instructions for OpenAI
messages = []

# System instructions for the assistant's personality and tone
system_msg = """Your name is Aria. You are the warm, cheerful, and efficient AI receptionist for "InspireWorks". Your voice should sound professional yet welcoming, like a helpful front-desk executive.

Your primary goal is to guide callers through a menu selection process. You must strictly follow this conversation flow:

1.  **Greeting:**
    * Check the current time in IST (Indian Standard Time).
    * If morning (00:00-11:59 IST), say "Good Morning".
    * If afternoon (12:00-16:00 IST), say "Good Afternoon".
    * If evening (16:00-23:59 IST), say "Good Evening".
    * Immediately follow the greeting with: "...and welcome to InspireWorks. Empowering your digital future."

2.  **Language Selection (Level 1):**
    * You must present the menu options exactly as follows:
    * Say in English: "Press 1 for English."
    * Immediately switch to Spanish and say: "Para Español, presione el número 2."

3.  **Handling Selection (Level 2):**
    * **If the user presses or says '1' (English):**
        * Acknowledge warmly: "Great, you've selected English."
        * Present the next options: "Press 1 to listen to our audio message, or press 2 to speak with a live associate."
    * **If the user presses or says '2' (Spanish):**
        * Switch to speaking Spanish.
        * Acknowledge: "Entendido, ha seleccionado Español."
        * Present the next options in Spanish: "Presione el 1 para escuchar nuestro mensaje de audio, o el 2 para hablar con un asociado."

4.  **Final Action:**
    * If they choose '1' (Audio), say: "Playing the audio message now."
    * If they choose '2' (Associate), say: "Connecting you now. Please hold."

Maintain a patient tone. If the user does not respond or the input is unclear, politely ask them to repeat the option."""

# Add system message to conversation history
messages.append({"role": "system", "content": system_msg})


# Load config (to avoid repetition in case of reloading config later)
def load_config(file_path='config.json'):
    global CONFIG
    with open(file_path, 'r') as f:
        CONFIG = json.load(f)


# Transcribes audio to text using Deepgram API
async def transcribe_audio(audio_chunk, channels=1, sample_width=2, frame_rate=8000):
    try:
        # Convert audio chunk into a NumPy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

        # Create an in-memory BytesIO object for WAV file format
        wav_io = io.BytesIO()

        # Write audio data as WAV format into BytesIO
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(frame_rate)
            wav_file.writeframes(audio_data.tobytes())

        # Reset the stream position of the in-memory WAV file
        wav_io.seek(0)

        # Send the audio to Deepgram for transcription
        response = await dg_client.transcription.prerecorded({
            'buffer': wav_io,  # Audio data in bytearray format
            'mimetype': 'audio/wav'
        }, {
            'punctuate': True  # Enables punctuation in transcription
        })

        # Extract the transcription result from the response
        transcription = response['results']['channels'][0]['alternatives'][0]['transcript']

        if transcription != '':
            print("Transcription: ", transcription)

        return transcription
    except Exception as e:
        print("An error occurred during transcription:")
        traceback.print_exc()


# Generate a response using OpenAI and send the reply via Plivo WebSocket
async def generate_response(input_text, plivo_ws):
    # Append user input to the conversation history
    messages.append({"role": "user", "content": input_text})

    # Call the OpenAI API to generate a conversational response
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    # Extract the assistant's response from the API call
    reply = response.choices[0].message.content

    # Append assistant response to the conversation history
    messages.append({"role": "assistant", "content": reply})

    print("OpenAI Response: ", reply)

    # Convert the response into speech and send via WebSocket
    await text_to_speech_file(reply, plivo_ws)


# Converts text to speech using ElevenLabs API and sends it via Plivo WebSocket
async def text_to_speech_file(text: str, plivo_ws):
    # Call ElevenLabs API to convert text into speech
    response = tts_client.text_to_speech.convert(
        voice_id="XrExE9yKIg1WjnnlVkGX",  # Using a pre-made voice (Adam)
        output_format="ulaw_8000",  # 8kHz audio format
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Collect the audio data from the response
    output = bytearray(b'')
    for chunk in response:
        if chunk:
            output.extend(chunk)

    # Encode the audio data in Base64 format
    encode = base64.b64encode(output).decode('utf-8')

    # Send the audio data via WebSocket to Plivo
    await plivo_ws.send(json.dumps({
        "event": "playAudio",
        "media": {
            "contentType": "audio/x-mulaw",
            "sampleRate": 8000,
            "payload": encode
        }
    }))


# Handles Plivo WebSocket communication for receiving and processing audio
async def plivo_handler(plivo_ws):
    async def plivo_receiver(plivo_ws, sample_rate=8000, silence_threshold=0.5):
        print('Plivo receiver started')

        # Initialize voice activity detection (VAD) with sensitivity level
        vad = webrtcvad.Vad(1)  # Level 1 is least sensitive

        inbuffer = bytearray(b'')  # Buffer to hold received audio chunks
        silence_start = 0  # Track when silence begins
        chunk = None  # Audio chunk

        try:
            async for message in plivo_ws:
                try:
                    # Decode incoming messages from the WebSocket
                    data = json.loads(message)

                    # If 'media' event, process the audio chunk
                    if data['event'] == 'media':
                        media = data['media']
                        chunk = base64.b64decode(media['payload'])
                        inbuffer.extend(chunk)

                    # If 'stop' event, end receiving process
                    if data['event'] == 'stop':
                        break

                    if chunk is None:
                        continue

                    # Check if the chunk contains speech or silence
                    is_speech = vad.is_speech(chunk, sample_rate)

                    if not is_speech:  # Detected silence
                        silence_start += 0.2  # Increment silence duration (200ms steps)
                        if silence_start >= silence_threshold:  # If silence exceeds threshold
                            if len(inbuffer) > 2048:  # Process buffered audio if large enough
                                transcription = await transcribe_audio(inbuffer)
                                if transcription != '':
                                    await generate_response(transcription, plivo_ws)
                            inbuffer = bytearray(b'')  # Clear buffer after processing
                            silence_start = 0  # Reset silence timer
                    else:
                        silence_start = 0  # Reset if speech is detected
                except Exception as e:
                    print(f"Error processing message: {e}")
                    traceback.print_exc()
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Websocket connection closed")
        except Exception as e:
            print(f"Error processing message: {e}")
            traceback.print_exc()

    # Start the receiver for WebSocket messages
    await plivo_receiver(plivo_ws)


# Router to handle incoming WebSocket connections and routes them to plivo_handler
async def router(websocket, path):
    if path == '/stream':
        print('Plivo connection incoming')
        await plivo_handler(websocket)


# Main function to start the WebSocket server
def main():
    # Start the WebSocket server on localhost port 5000
    server = websockets.serve(router, 'localhost', 5000)

    # Run the event loop for the WebSocket server
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


# Entry point for running the script
if __name__ == '__main__':
    sys.exit(main() or 0)
