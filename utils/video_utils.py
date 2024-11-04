import os
import cv2
import moviepy.editor as mp
import io
# import ffmpeg
from pydub import AudioSegment

from PIL import Image

from openai import OpenAI


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def transcribe_audio(video_path):
    client = OpenAI()
    video = mp.VideoFileClip(video_path)

    audio_temp_path = "temp_audio.wav"
    audio = video.audio
    audio_duration = audio.duration
    audio = audio.subclip(0, min(10, audio_duration))
    
    audio.write_audiofile(audio_temp_path, verbose=False, logger=None)

    audio_segment = AudioSegment.from_file(audio_temp_path, format="wav")

    audio_buffer = io.BytesIO()
    audio_segment.export(audio_buffer, format="wav")
    audio_buffer.name = "audio.wav"
    audio_buffer.seek(0)

    os.remove(audio_temp_path)

    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_buffer,
        response_format="text"
        )
    
    transcription = check_transription_validity(transcription)
    # print(transcription)
    return transcription

def check_transription_validity(transcription):
    client = OpenAI()
    prompt = '''
    You are an assistant responsible for evaluating the validity and meaningfulness of transcriptions. A transcription is considered meaningful if it provides insights into the user's activities or discusses information that cannot be easily inferred from the visual content alone. Fixed typo and output the transription. Otherwise, return empty.
    '''
    # prompt = 'You are an assistant responsible for evaluating whether the transcription is valid and meaningful. If not return empty. Otherwise, output the transcription (typo-free).'

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcription}
        ]
        )
    result = completion.choices[0].message.content
    return result

def sample_frames_from_video(video_path, num_frames):
    # video = mp.VideoFileClip(video_path)
    # duration = video.duration
    # if duration > 10:
    #     video = video.subclip(0, 10)
    frames = []
    
    # total_frames = int(video.fps * video.duration)
    # frames_indices = [int(i) for i in range(0, total_frames, total_frames//num_frames)]
    # for idx in frames_indices:
    #     frame = video.get_frame(idx/video.fps)
    #     frame = Image.fromarray(frame)
    #     frames.append(frame)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1

    frames_indices = [int(i) for i in range(0, total_frames, round(total_frames/num_frames))]
    for idx in frames_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    return frames