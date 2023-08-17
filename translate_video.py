import moviepy.editor as mpy
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
from pydub import AudioSegment
from deepmultilingualpunctuation import PunctuationModel
from nltk.tokenize import sent_tokenize

import requests
import io

print("running")
# Initialize the Google Cloud clients
speech_client = speech.SpeechClient()
translate_client = translate.Client()
tts_client = texttospeech.TextToSpeechClient()

# 1. Extract audio from the video
video = mpy.VideoFileClip("input_video.mp4")
video.audio.write_audiofile("stereo_audio.wav")

# Convert stereo audio to mono using pydub
stereo_audio = AudioSegment.from_wav("stereo_audio.wav")
mono_audio = stereo_audio.set_channels(1)
mono_audio.export("temp_audio.wav", format="wav")

# 2. Transcribe the audio from English to text
with open("temp_audio.wav", "rb") as audio_file:
    response = speech_client.recognize(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
            enable_word_time_offsets=True  # Enable word-level timestamps
        ),
        audio=speech.RecognitionAudio(content=audio_file.read()),
    )

model = PunctuationModel()
words = [(word_info.word, word_info.end_time) for result in response.results for word_info in result.alternatives[0].words]
print(words)

transcription = ' '.join([word[0] for word in words])
punctuated_transcription = model.restore_punctuation(transcription)
print(punctuated_transcription)

# Break the punctuated transcription into sentences
sentences = sent_tokenize(punctuated_transcription)

punctuated_words = punctuated_transcription.split()
print(punctuated_words)
# Initialize variables
word_idx = 0
sentence_with_times = []

for sentence in sentences:
    sentence_words = sentence.split()
    last_word = sentence_words[-1][:-1]
    
    while word_idx < len(punctuated_words):
        word_info = response.results[0].alternatives[0].words[word_idx]
        if word_info.word == last_word:
            end_time = word_info.end_time.seconds + word_info.end_time.microseconds * 1e-6
            sentence_with_times.append((sentence, end_time))
            word_idx += 1
            break
        word_idx += 1

    if word_idx >= len(punctuated_words):
        break

print(sentence_with_times)

def change_voice(transcription, file_name):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/onwK4e9ZLuTAKqWW03F9"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "<key>"
    }

    data = {
        "text": transcription,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }


    response = requests.post(url, headers=headers, json=data, stream=True)
    output_filename = file_name
    with open(output_filename, "wb") as output:
        output.write(response.content)

    return response.content

# Create audio segments based on sentence boundaries
last_end_time = 0
audio_segments = []

for sentence, end_time in sentence_with_times:
    audio_filename = f"audio_{end_time}.mp3"
    audio_content=change_voice(sentence, audio_filename)
    with open(audio_filename, "wb") as out:
        out.write(audio_content)
    
    audio_clip = mpy.AudioFileClip(audio_filename)
    # Extract a segment of the original audio from last_end_time to the start of the new audio clip
    original_segment = video.audio.subclip(last_end_time, end_time - audio_clip.duration)
    
    # Append the original segment and the new audio clip to the list
    audio_segments.extend([original_segment, audio_clip])
    
    # Update the last_end_time
    last_end_time = end_time

# Add remaining audio after the last end_time
audio_segments.append(video.audio.subclip(last_end_time))

# Combine all audio segments
modified_audio = mpy.concatenate_audioclips(audio_segments)

# Attach the modified audio track to the video
final_video = video.set_audio(modified_audio)
final_video.write_videofile("output_video.mp4")

print("Conversion complete!")

    # # 5. Replace the original audio in the video with the new English audio
    # english_audio = mpy.AudioFileClip("output.mp3")
    # final_video = video.set_audio(english_audio)
    # final_video.write_videofile("output_video.mp4")

    # print("Conversion complete!")