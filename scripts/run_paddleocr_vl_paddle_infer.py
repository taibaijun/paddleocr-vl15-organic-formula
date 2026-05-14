import argparse
import json
from pathlib import Path

import numpy as np
import paddle
from PIL import Image
from paddleformers.generation import GenerationConfig
from paddleformers.transformers.feature_extraction_utils import BatchFeature
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.transformers.paddleocr_vl.processor import PaddleOCRVLProcessorKwargs


PROMPTS = {
    "ocr": "OCR:",
    "chemical": "Chemical Structure Recognition:",
    "formula": "Formula Recognition:",
}


def patch_processor_call(processor) -> None:
    def fixed_call(self, images=None, text=None, videos=None, **kwargs):
        output_kwargs = self._merge_kwargs(
            PaddleOCRVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError("fps should be a single number or match video_grid_thw length.")
            videos_inputs.update({"second_per_grid_ts": paddle.tensor(second_per_grid_ts)})
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        def placeholder_count(grid_item) -> int:
            if hasattr(grid_item, "numpy"):
                values = grid_item.numpy()
            else:
                values = np.asarray(grid_item)
            return int(np.prod(values) // self.image_processor.merge_size // self.image_processor.merge_size)

        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * placeholder_count(image_grid_thw[index]),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * placeholder_count(video_grid_thw[index]),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    processor.__class__.__call__ = fixed_call


def run_one(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    official_template: bool,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if official_template:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pd",
        )
    else:
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=prompt_text,
            images=[image],
            return_dict=True,
            return_tensors="pd",
        )
    bos_token_id = getattr(processor.tokenizer, "bos_token_id", None)
    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(model.config, "bos_token_id", 0)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", 1)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", 0)
    generation_config = GenerationConfig(
        do_sample=False,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True,
    )
    with paddle.no_grad():
        generated_ids = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
    ids = generated_ids[0].tolist()[0]
    return processor.decode(ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL inference through PaddleFormers.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Optional unmerged LoRA adapter directory to load on top of --model-path.",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--official-template",
        action="store_true",
        help="Use the tokenize=True apply_chat_template path shown in PaddleFormers examples.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(PROMPTS),
        default=["ocr", "chemical", "formula"],
    )
    args = parser.parse_args()

    paddle.set_device("gpu:0")
    model = AutoModelForConditionalGeneration.from_pretrained(
        args.model_path,
        convert_from_hf=True,
    ).eval()
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    if args.lora_path:
        from paddleformers.peft import LoRAModel

        lora_model = LoRAModel.from_pretrained(model, args.lora_path)
        model = lora_model.model.eval()

    processor = AutoProcessor.from_pretrained(args.lora_path or args.model_path)
    patch_processor_call(processor)

    image = Image.open(args.image).convert("RGB")
    results = {}
    for task in args.tasks:
        prompt = PROMPTS[task]
        text = run_one(model, processor, image, prompt, args.max_new_tokens, args.official_template)
        results[task] = {"prompt": prompt, "text": text}
        print(f"===== {task} / {prompt} =====")
        print(text)
        print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

