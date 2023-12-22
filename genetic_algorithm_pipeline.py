import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from data_pipeline import get_processed_data


# Load the dataframes
qa_data, prompt_df = get_processed_data()

# Extract the first question and answer from qa_data
question, expected_answer = qa_data.iloc[0][['Question', 'Answer']]

# Load the model and tokenizer
model_name = "mistralai" # Set the directory where the llm model is stored
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Assuming 'phrases' is a list of all available phrases from prompt_df
phrases = list(prompt_df['Phrase'])

# Define min_perplexity and max_perplexity based on your data or set arbitrary values
min_perplexity, max_perplexity = 1, 100  # Adjust as needed

# Function to create initial population
def create_initial_population(size, num_phrases):
    return [[random.choice(phrases) for _ in range(num_phrases)] for _ in range(size)]

# Crossover function
def crossover(parent1, parent2):
    child = []
    split1, split2 = random.randint(1, len(parent1)), random.randint(1, len(parent2))
    child.extend(parent1[:split1])
    child.extend(parent2[:split2])
    return child

# Mutation function
def mutate(parent):
    if parent:
        mutation_index = random.randint(0, len(parent) - 1)
        parent[mutation_index] = random.choice(phrases)

# Insertion function
def insert(parent):
    if len(parent) < 6:
        parent.insert(random.randint(0, len(parent)), random.choice(phrases))

# Deletion function
def delete(parent):
    if parent:
        del parent[random.randint(0, len(parent) - 1)]

# Limit the number of phrases to between 0 and 6
def limit_phrases(parent):
    return parent[:6]

def evaluate_fitness(individual):
    # Combine the question with the individual prompt
    prompt = ''.join(individual) + ' ' + question

    # Get model response and perplexity
    response, perplexity = get_model_response_and_perplexity(prompt)

    # Calculate accuracy score
    accuracy_score = 1 if expected_answer in response else 0

    # Calculate normalized perplexity
    normalized_perplexity = (perplexity - min_perplexity) / (max_perplexity - min_perplexity)

    # Calculate fitness score
    fitness_score = accuracy_score - normalized_perplexity

    return fitness_score

def get_model_response_and_perplexity(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    # Flatten the output to a 1D list
    logits_flat = outputs.logits.argmax(dim=-1).view(-1).tolist()
    # Decode the flattened list of token IDs
    response = tokenizer.decode(logits_flat, skip_special_tokens=True)
    return response, perplexity