from paddleformers.transformers import AutoProcessor


for model_path in [
    "PaddlePaddle/PaddleOCR-VL-1.5",
    "output/paddleocr_vl15_caption_plain_lora_safe50/export",
]:
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    print("PATH", model_path)
    print(
        "IDS",
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    )
    print(
        "TOKENS",
        getattr(tokenizer, "bos_token", None),
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
    )

