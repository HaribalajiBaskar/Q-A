# from happytransformer import HappyTextSummarization
#
# happy_ts = HappyTextSummarization()
#
#
# def summarize_text(text):
#     summary = happy_ts.summarize_text(text)
#     return summary

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize the T5 model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def summarize_text(text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs.input_ids,
                                 attention_mask=inputs.attention_mask,
                                 max_length=150,
                                 min_length=30,
                                 num_beams=2,
                                 no_repeat_ngram_size=3)

    # Decode the summary tokens and convert to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
