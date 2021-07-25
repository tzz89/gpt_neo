import streamlit as st
from transformers import GPTNeoForCausalLM, GPT2Tokenizer


@st.cache(allow_output_mutation=True)
def load_tokenizer_model():
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    return model, tokenizer



if __name__ == "__main__":
    st.write("""
    # Hello Soon Siang

    This is a sample of gpt-neo generation

    """)

    model, tokenizer = load_tokenizer_model()


    with st.form(key='main_form'):
        text_input = st.text_area('Enter your text here')
        max_length = st.number_input("Max generated length (between 10 to 100), the longer the length, the longer the generation time", min_value=10, max_value=100, value=20, step=1)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        input_ids = tokenizer(text_input, return_tensors="pt").input_ids
        gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=max_length)
        gen_text = tokenizer.batch_decode(gen_tokens)[0]
        st.write("# Generated text")
        st.text(gen_text)

