# Project Title:
Automated Prompt Engineering Using Mistral.ai

## Use Case:
This project utilizes a genetic algorithm alongside a language model to process and generate optimized text prompts based on a question and answer dataset.

It's 'fitness function' can be use to evaluate the accuracy & relevance of the outputs produced from generated prompts.

### Example: 

For example, when the program is provided with a topic ("Cooking") & and a desired output template, a set number of prompts is generated. When each prompt produces an output, the output's effectiveness is evaluated by the "fitness function" within the program. After a set number of iterations, the prompt that produces the most "fit" output is selected & returned to the user.

This program was completed as a submission towards the Openmesh Open D/I Hackathon challenge in December 2023.

The Hackathon Challenge has since concluded.

## Installation instructions.

Before running the application, ensure you have Python >3.9 installed. You can download and install Python from here.

## Dependencies
Install the required dependencies by running:
pip install -r requirements.txt

## Local Storage of LLM Model Files
The project uses a local instance of the mistralai language model. Ensure that the mistralai folder is present at the root of the project and contains all the necessary model files. Available at: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1 (Will also run using Mistral-7b-v0.1 (https://huggingface.co/mistralai/Mistral-7B-v0.1))

## Running the Application
Data Pipeline
The data_pipeline.py script is responsible for loading and preprocessing the question and answer datasets. Ensure the datasets are correctly placed in the input/qa_dataset/ directory.

Genetic Algorithm Pipeline
The genetic_algorithm_pipeline.py script contains the logic for the genetic algorithm and interaction with the language model.

This will run locally by running app.py, 

alternatively-

## Flask Server
To start the Flask server, run:
python server.py
This will start the server and make the API endpoints available.

## API Endpoints
GET /get_data: Retrieves data from the server.
POST /post_data: Accepts data and updates the server state.
Usage
Describe how to use the application, including example requests to the Flask server and how to interpret the responses.

### Known issues
- The program does take a long time to execute. Due to our team members pursuing other fields, this issue remains un-optimized.

