import librosa
import numpy as np
import cv2
import os

def robust_load(file, sr=22050):
    """
    Load audio using librosa, with a robust fallback to PyAV for formats 
    that might require system ffmpeg (like WebM from st.audio_input).
    """
    try:
        # Standard librosa load (uses soundfile or audioread)
        return librosa.load(file, sr=sr)
    except Exception as e:
        print(f"DEBUG: librosa.load failed: {e}. Falling back to PyAV.")
        try:
            import av
            container = av.open(file)
            # Find the first audio stream
            stream = next(s for s in container.streams if s.type == 'audio')
            
            # Setup resampler to target sample rate, mono, float32
            resampler = av.audio.resampler.AudioResampler(
                format='fltp',
                layout='mono',
                rate=sr
            )
            
            audio_frames = []
            for frame in container.decode(stream):
                resampled_frames = resampler.resample(frame)
                for rf in resampled_frames:
                    # Convert to ndarray and flatten (resampler output is mono/fltp)
                    audio_frames.append(rf.to_ndarray().flatten())
            
            if not audio_frames:
                raise ValueError("No audio frames could be decoded via PyAV")
            
            full_audio = np.concatenate(audio_frames)
            return full_audio, sr
        except Exception as av_e:
            print(f"DEBUG: PyAV fallback also failed: {av_e}")
            raise av_e

def extract_features(file):
    audio, sr = robust_load(file, sr=22050)

    # 180 feature vector
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)

    vec = np.hstack([
        np.mean(mfcc.T,axis=0),
        np.mean(chroma.T,axis=0),
        np.mean(mel.T,axis=0)
    ])

    vec = vec[:180]
    vec = vec.reshape(1,180)

    # spectrogram for ShEMO
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=120)
    mel_db = librosa.power_to_db(mel_spec)
    mel_db = cv2.resize(mel_db,(130,120))
    mel_db = mel_db.reshape(1,120,130,1)

    return vec, mel_db
