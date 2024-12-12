import logging
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Streamlit page configuration
st.set_page_config(page_title="Llama-2 QnA", layout="wide")

# Initialize logging to suppress warnings
logging.basicConfig(level=logging.CRITICAL)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    base_model_id = "NousResearch/Llama-2-7b-chat-hf"
    peft_model_id = "ShahzaibDev/Llama2-7B-Qna"

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Load the base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=quantization_config
    )

    # Load the fine-tuned PEFT model
    model = PeftModel.from_pretrained(base_model, peft_model_id)

    # Switch the model to evaluation mode
    model.eval()

    return model, tokenizer

# Load the model and tokenizer
with st.spinner("Loading model..."):
    model, tokenizer = load_model_and_tokenizer()

# Streamlit UI
st.title("Llama-2 QnA")
st.markdown("This app answers your General questions using a fine-tuned Llama-2 model.")

# User input
prompt = st.text_area("Enter your question:", placeholder="What is generative AI?", height=150)

# Generate response button
if st.button("Generate Response"):
    if not prompt.strip():
        st.error("Please enter a valid question.")
    else:
        formatted_prompt = f"<s>[INST] {prompt.strip()} [/INST]"
        
        # Tokenize the input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate the response
        with st.spinner("Generating response..."):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50
                )

            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the response
        st.subheader("Response:")
        st.write(response)

# Footer
st.markdown("Powered by *Llama-2* and PEFT fine-tuning.")