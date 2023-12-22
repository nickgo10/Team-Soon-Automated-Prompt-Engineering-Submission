# Imports
import pandas as pd
import string

def get_processed_data():

    # Load question and answer dataset into dataframes
    df1 = pd.read_csv('input/qa_dataset/S08_question_answer_pairs.txt', sep='\t')
    df2 = pd.read_csv('input/qa_dataset/S09_question_answer_pairs.txt', sep='\t')
    df3 = pd.read_csv('input/qa_dataset/S10_question_answer_pairs.txt', sep='\t', encoding = 'ISO-8859-1')

    # Combine into one dataframe
    qa_data = pd.concat([df1, df2, df3], ignore_index=True)

    #Join topic (article title) to the question to keep context
    qa_data['Question'] = qa_data['ArticleTitle'].str.replace('_', ' ') + ' ' + qa_data['Question']
    qa_data = qa_data[['Question', 'Answer']]

    #Delete duplicates
    qa_data = qa_data.drop_duplicates(subset='Question')

    #Clean the text
    def text_cleaning(text):
        if isinstance(text, str):  # Check if the value is a string
            text = "".join([char for char in text if char not in string.punctuation])
        return text

    # Apply the text cleaning function to the "Questions" column, handling NaN values
    qa_data['Question'] = qa_data['Question'].apply(text_cleaning)

    #Drop Null vales
    qa_data = qa_data.dropna()



    #Load prompt data
    # File path
    file_path = 'input/ga_prompts/prompts.txt'

    # Initialize an empty list to store the phrases
    phrases = []

    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and check if the line is enclosed in quotation marks
            stripped_line = line.strip()
            if stripped_line.startswith('"') and stripped_line.endswith('"'):
                # Remove the quotation marks and add to the list
                phrases.append(stripped_line.strip('"'))

    # Create DataFrame from the list of phrases
    prompt_df = pd.DataFrame({'Phrase': phrases})

    return qa_data, prompt_df