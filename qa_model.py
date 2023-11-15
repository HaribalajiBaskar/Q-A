# # Function to ask a question and get a response using the Hugging Face pipeline
# from happytransformer import HappyQuestionAnswering
# def ask_question(question, context):
#         happy_qa = HappyQuestionAnswering()
#         answer = happy_qa.answer_question(context, question, top_k = 2)
#
#         return answer


from transformers import T5Tokenizer, T5ForConditionalGeneration

def ask_question(question, context):
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    # Prepare the input
    input_text = "question: " + question + " context: " + context

    # Tokenize the input
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate the answer
    output = model.generate(input_ids)

    # Decode and return the answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Convert single-word answer to a sentence
    if len(answer.split()) == 1:
        answer = answer.capitalize() + '.'

    return answer
