import sys
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer
from groq_caller import speechtoText

class AudioRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.is_recording = False
        self.sample_rate = 44100
        self.recording = []
        
    def initUI(self):
        layout = QVBoxLayout()
        
        self.recordButton = QPushButton('Start Recording', self)
        self.recordButton.clicked.connect(self.toggleRecording)
        layout.addWidget(self.recordButton)
        
        self.statusLabel = QLabel('Not recording', self)
        layout.addWidget(self.statusLabel)
        
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
        self.statusLabel.setText('Recording...')
        self.recording = sd.rec(int(5 * self.sample_rate), samplerate=self.sample_rate, channels=1)
        
    def stopRecording(self):
        self.is_recording = False
        self.recordButton.setText('Start Recording')
        self.statusLabel.setText('Not recording')
        sd.stop()
        self.saveRecording()
    
    def saveRecording(self):
        if len(self.recording) == 0:
            return
        
        wavfile.write('output.wav', self.sample_rate, self.recording)
        print("Recording saved as 'output.wav'")
        self.statusLabel.setText('Recording saved as output.wav')
        speechtoText()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioRecorder()
    sys.exit(app.exec_())