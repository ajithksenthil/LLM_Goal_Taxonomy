import openai
import time

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your API key

# Define your taxonomy sentences
taxonomy_sentences = [
    "How can I improve my writing skills?",
    "What is the best way to lose weight?",
    "Tips for learning a new language quickly.",
    "How to invest in the stock market?",
    # Add more sentences as needed
]

# Number of similar sentences to generate for each taxonomy sentence
num_similar_sentences = 5

# Function to generate similar sentences using OpenAI's GPT-3.5 Turbo
def generate_similar_sentences(prompt, num_sentences=5):
    # Construct the system message and user prompt
    system_message = "You are an assistant that generates sentences similar in meaning to the input sentence."
    user_message = f"Generate {num_sentences} sentences that are semantically similar to the following sentence:\n\"{prompt}\""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=500,
            n=1,
            stop=None,
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        # Split the generated text into individual sentences
        similar_sentences = [line.strip('- ').strip() for line in generated_text.split('\n') if line.strip()]
        return similar_sentences
    except openai.error.OpenAIError as e:
        print(f"Error generating sentences: {e}")
        return []

# For each sentence in the taxonomy, generate similar sentences
for sentence in taxonomy_sentences:
    print(f"\nTaxonomy Sentence: {sentence}")
    similar_sentences = generate_similar_sentences(sentence, num_similar_sentences)
    if similar_sentences:
        print("Similar Sentences:")
        for idx, sim_sentence in enumerate(similar_sentences, start=1):
            print(f"  {idx}. {sim_sentence}")
        # Introduce a short delay to comply with rate limits
        time.sleep(1)
    else:
        print("Failed to generate similar sentences.")
