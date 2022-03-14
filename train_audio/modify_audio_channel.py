from pydub import AudioSegment

sound = AudioSegment.from_wav("/home/kongshui/Documents/audio/2_channels/carpet_2.wav")
sound = sound.set_channels(1)
sound.export("/home/kongshui/Documents/audio/1_channels/carpet_input/carpet2.wav", format="wav")
