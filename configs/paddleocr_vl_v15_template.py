from paddleformers.datasets.template.template import *
from paddleformers.datasets.template.mm_plugin import *
from paddleformers.datasets.template.augment_utils import *


@dataclass
class PaddleOCRVLV15Plugin(BasePlugin):
    image_bos_token: str = "<|IMAGE_START|>"
    image_eos_token: str = "<|IMAGE_END|>"

    def __init__(self, image_token, video_token, audio_token, **kwargs):
        super().__init__(image_token, video_token, audio_token, **kwargs)
        self.image_augmentation = self.get_ocr_augmentations(
            rotation_p=0.0,
            jpeg_p=0.0,
            scale_p=0.0,
            padding_p=0.0,
            color_jitter_p=0.0,
        )

    def get_ocr_augmentations(
        self,
        scale_range=(0.8, 1.2),
        scale_p=0.5,
        padding_range=(0, 15),
        padding_p=0.5,
        rotation_degrees=[0],
        rotation_p=0.5,
        color_jitter_p=0.5,
        jpeg_quality_range=(40, 90),
        jpeg_p=0.5,
    ):
        augmentations = []

        if scale_p > 0:
            augmentations.append(RandomApply([RandomScale(scale_range=scale_range)], p=scale_p))

        if padding_p > 0:
            augmentations.append(
                RandomApply([RandomSingleSidePadding(padding_range=padding_range, fill="white")], p=padding_p)
            )

        if rotation_p > 0 and rotation_degrees:
            augmentations.append(
                RandomApply(
                    [RandomDiscreteRotation(degrees=rotation_degrees, interpolation="nearest", expand=True)],
                    p=rotation_p,
                )
            )

        if color_jitter_p > 0:
            augmentations.append(
                RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)], p=color_jitter_p)
            )

        if jpeg_p > 0:
            augmentations.append(RandomApply([JpegCompression(quality_range=jpeg_quality_range)], p=jpeg_p))

        return transforms.Compose(augmentations)

    @override
    def _preprocess_image(self, image, **kwargs):
        width, height = image.size
        image_max_pixels = kwargs["image_max_pixels"]
        image_min_pixels = kwargs["image_min_pixels"]
        image_processor = kwargs["image_processor"]

        resized_height, resized_width = image_processor.get_smarted_resize(
            height,
            width,
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
        )[0]

        image = image.resize((resized_width, resized_height))

        if image and hasattr(self, "image_augmentation"):
            image = self.image_augmentation(image)

        return image

    @override
    def _get_mm_inputs(
        self,
        images,
        videos,
        audios,
        processor,
        **kwargs,
    ):
        image_processor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(image_processor, "max_pixels", 1003520),
                image_min_pixels=getattr(image_processor, "min_pixels", 112896),
                image_processor=image_processor,
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pd"))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages,
        images,
        videos,
        audios,
        mm_inputs,
        processor,
    ):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        image_processor = getattr(processor, "image_processor")

        merge_length = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod().item() // merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.image_bos_token}{self.image_token * image_seqlen}{self.image_eos_token}",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


register_mm_plugin(
    name="paddleocr_vl_v15",
    plugin_class=PaddleOCRVLV15Plugin,
)

register_template(
    name="paddleocr_vl_v15",
    format_user=StringFormatter(slots=["User: {{content}}\nAssistant:\n"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_prefix=EmptyFormatter(slots=["<|begin_of_sentence|>"]),
    chat_sep="<|end_of_sentence|>",
    mm_plugin=get_mm_plugin(name="paddleocr_vl_v15", image_token="<|IMAGE_PLACEHOLDER|>"),
)

