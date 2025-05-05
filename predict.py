from cog import BasePredictor, Input, Path
import os
import shutil

from infer import load_models, main

WAN_MODEL_DIR = "./models/Wan2.1-I2V-14B-720P"
FANTASYTALKING_MODEL_PATH = "./models/fantasytalking_model.ckpt"
WAV2VEC_MODEL_DIR = "./models/wav2vec2-base-960h"

OUTPUT_DIR = "./output"
INPUT_DIR = "./input"


class Arguments:
    def __init__(
        self,
        image=None,
        audio=None,
        prompt=None,
        image_size=None,
        audio_scale=None,
        prompt_cfg_scale=None,
        audio_cfg_scale=None,
        max_num_frames=None,
        fps=None,
        num_persistent_param_in_dit=None,
        seed=None,
    ):
        self.wan_model_dir = WAN_MODEL_DIR
        self.fantasytalking_model_path = FANTASYTALKING_MODEL_PATH
        self.wav2vec_model_dir = WAV2VEC_MODEL_DIR
        self.output_dir = OUTPUT_DIR
        self.image_path = image
        self.audio_path = audio
        self.prompt = prompt
        self.image_size = image_size
        self.audio_scale = audio_scale
        self.prompt_cfg_scale = prompt_cfg_scale
        self.audio_cfg_scale = audio_cfg_scale
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.num_persistent_param_in_dit = num_persistent_param_in_dit
        self.seed = seed


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        args = Arguments(
            num_persistent_param_in_dit=getattr(
                self, "num_persistent_param_in_dit", None
            )
        )
        _pipe, _fantasytalking, _wav2vec_processor, _wav2vec = load_models(args)
        self.pipe = _pipe
        self.fantasytalking = _fantasytalking
        self.want2vec_processor = _wav2vec_processor
        self.wav2vec = _wav2vec

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # The arguments and types the model takes as input
    def predict(
        self,
        image: Path = Input(description="input image", default=None),
        audio: Path = Input(description="input audio", default=None),
        prompt: str = Input(description="prompt", default="A woman is talking."),
        image_size: int = Input(
            description="The image will be resized proportionally to this size.",
            default=512,
        ),
        audio_scale: float = Input(
            description="Audio condition injection weight", default=1.0
        ),
        prompt_cfg_scale: float = Input(description="Prompt cfg scale", default=5.0),
        audio_cfg_scale: float = Input(description="Audio cfg scale", default=5.0),
        max_num_frames: int = Input(
            description="The maximum frames for generating videos, the audio part exceeding max_num_frames/fps will be truncated.",
            default=81,
        ),
        fps: int = Input(
            description="The frame rate of the generated video, for fast talking audio, fps should be increased for smooth lips sync",
            default=23,
        ),
        num_persistent_param_in_dit: int = Input(
            description="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
            default=None,
        ),
        seed: int = Input(description="Random seed", default=1111),
    ) -> Path:
        for directory in [OUTPUT_DIR, INPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        """Run a single prediction on the model"""
        if num_persistent_param_in_dit != None:
            self.num_persistent_param_in_dit = num_persistent_param_in_dit
            self.setup()

        image_filename = self.filename_with_extension(image, "image")
        self.handle_input_file(image, image_filename)

        audio_filename = self.filename_with_extension(audio, "audio")
        self.handle_input_file(audio, audio_filename)

        args = Arguments(
            image=image_filename,
            audio=audio_filename,
            prompt=prompt,
            image_size=image_size,
            audio_scale=audio_scale,
            prompt_cfg_scale=prompt_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            max_num_frames=max_num_frames,
            fps=fps,
            num_persistent_param_in_dit=num_persistent_param_in_dit,
            seed=seed,
        )

        output = main(
            args, self.pipe, self.fantasytalking, self.wav2vec_processor, self.wav2vec
        )
        return output
