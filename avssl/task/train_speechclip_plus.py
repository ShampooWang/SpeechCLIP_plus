from ..model import SpeechCLIP_plus
from .base_task import BaseTask, TrainSpeechClipBaseTask


class TrainSpeechCLIP_plus(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(SpeechCLIP_plus)
