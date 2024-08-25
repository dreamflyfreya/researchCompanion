import sys
import pyaudio
import wave
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import QTimer
import os
from groq_caller import speechtoText
# Initialize the Groq client

class AudioRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        
    def initUI(self):
        layout = QVBoxLayout()
        
        self.recordButton = QPushButton('Start Recording', self)
        self.recordButton.clicked.connect(self.toggleRecording)
        layout.addWidget(self.recordButton)
        
        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Audio Recorder')
        self.show()
        
    def toggleRecording(self):
        if not self.is_recording:
            self.startRecording()
        else:
            self.stopRecording()
    
    def startRecording(self):
        self.is_recording = True
        self.recordButton.setText('Stop Recording')
        
        self.frames = []
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024,
                                      stream_callback=self.callback)
        self.stream.start_stream()
    
    def stopRecording(self):
        self.is_recording = False
        self.recordButton.setText('Start Recording')
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.saveRecording()
        # call groq
        speechtoText()
    
    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def saveRecording(self):
        if not self.frames:
            return
        
        wf = wave.open('tem_output.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print("Recording saved as 'output.wav'")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioRecorder()
    sys.exit(app.exec_())