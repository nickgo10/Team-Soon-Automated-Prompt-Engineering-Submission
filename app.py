#!/usr/bin/env python3
import pandas as pd
import random
from data_pipeline import get_processed_data
from genetic_algorithm_pipeline import create_initial_population, crossover, mutate, insert, delete, limit_phrases, evaluate_fitness

# Load the dataframes
qa_data, prompt_df = get_processed_data()

# Example usage

# Extract the first question and answer from qa_data
question, expected_answer = qa_data.iloc[0][['Question', 'Answer']]

population_size = 100
num_phrases_per_parent = 3
num_generations = 10

# Create initial population
population = create_initial_population(population_size, num_phrases_per_parent)

# Open a file to record the top 10 prompts for each generation
with open("top_prompts_per_generation.txt", "w") as file:

    # Run the genetic algorithm for 10 generations
    for generation in range(num_generations):
        # Create new children and add to population
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)

            # Apply genetic operations
            if random.random() < 0.5:  # Mutation probability
                mutate(child)
            if random.random() < 0.2:  # Insertion probability
                insert(child)
            if random.random() < 0.2:  # Deletion probability
                delete(child)

            child = limit_phrases(child)
            population.append(child)

        # Introduce new random individuals for diversity
        num_new_individuals = int(population_size * 0.1)  # e.g., 10% of the population size
        new_individuals = create_initial_population(num_new_individuals, num_phrases_per_parent)
        population.extend(new_individuals)

        # Evaluate fitness of each individual
        fitness_scores = [evaluate_fitness(individual) for individual in population]

        # Combine individuals and their fitness scores
        population_with_fitness = list(zip(population, fitness_scores))

        # Sort by fitness score
        population_sorted = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)

        # Select the top 10 individuals
        top_individuals = population_sorted[:10]

        # Write the top 10 individuals of this generation to the file
        file.write(f"Generation {generation + 1}:\n")
        for i, (individual, fitness) in enumerate(top_individuals, 1):
            prompt = ' '.join(individual)
            file.write(f"  {i}. Prompt: {prompt}\n      Fitness Score: {fitness}\n")
        file.write("\n")
        print("Top 10 prompts for each generation have been saved to 'top_prompts_per_generation.txt'")

        # Select the top-performing individuals for the next generation
        selection_count = int(len(population) * 0.5)
        next_generation = [individual for individual, _ in population_sorted[:selection_count]]

        # Update population for the next generation
        population = next_generation + random.sample(new_individuals, len(new_individuals) // 2)

        # Logging the progress
        print(f"Generation {generation + 1} completed. Population size: {len(population)}")







        # # Load the model and tokenizer
# model_name = "mistralai/Mistral-7B-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Assuming 'phrases' is a list of all available phrases from prompt_df
# phrases = list(prompt_df['Phrase'])

# # Define min_perplexity and max_perplexity based on your data or set arbitrary values
# min_perplexity, max_perplexity = 1, 100  # Adjust as needed

# # Function to create initial population
# def create_initial_population(size, num_phrases):
#     return [[random.choice(phrases) for _ in range(num_phrases)] for _ in range(size)]

# # Crossover function
# def crossover(parent1, parent2):
#     child = []
#     split1, split2 = random.randint(1, len(parent1)), random.randint(1, len(parent2))
#     child.extend(parent1[:split1])
#     child.extend(parent2[:split2])
#     return child

# # Mutation function
# def mutate(parent):
#     if parent:
#         mutation_index = random.randint(0, len(parent) - 1)
#         parent[mutation_index] = random.choice(phrases)

# # Insertion function
# def insert(parent):
#     if len(parent) < 6:
    """_summary_
    """#         parent.insert(random.randint(0, len(parent)), random.choice(phrases))

# # Deletion function
# def delete(parent):
#     if parent:
#         del parent[random.randint(0, len(parent) - 1)]

# # Limit the number of phrases to between 0 and 6
# def limit_phrases(parent):
#     return parent[:6]

# def evaluate_fitness(query, expected_answer):
#     response, perplexity = get_model_response_and_perplexity(query)
#     accuracy_score = 1 if expected_answer in response else 0
#     normalized_perplexity = (perplexity - min_perplexity) / (max_perplexity - min_perplexity)
#     w1, w2 = 0.7, 0.3
#     fitness_score = w1 * accuracy_score - w2 * normalized_perplexity
#     return fitness_score

# def get_model_response_and_perplexity(input_text):
#     inputs = tokenizer(input_text, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs, labels=inputs["input_ids"])
#     loss = outputs.loss
#     perplexity = torch.exp(loss).item()
#     response = tokenizer.decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
#     return response, perplexity