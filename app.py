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