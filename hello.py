import streamlit as st
import pandas as pd
import transformers
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Title of the app
st.title("BERT Fine Tuning for Question Answering")

# Initialize the tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to find start and end positions of the answer
def find_answer_positions(context, answer):
    start = context.find(answer)
    if start == -1:
        return 0, 0  # Return default values if answer not found
    end = start + len(answer) - 1
    return start, end

# Tokenization function 
def tokenize_function(examples):
    # Using 'Question' as context here for the sake of example
    tokenized_inputs = tokenizer(
        examples['Question'], examples['Answer'],  # Use Question and Answer for now
        padding='max_length', truncation=True,
        return_tensors='pt'
    )
    
    start_positions = []
    end_positions = []
    
    for question, answer in zip(examples['Question'], examples['Answer']):
        # Assuming the question itself is the context
        start, end = find_answer_positions(question, answer)
        start_positions.append(start)
        end_positions.append(end)
    
    tokenized_inputs['start_positions'] = start_positions
    tokenized_inputs['end_positions'] = end_positions
    return tokenized_inputs

def main():
    # File uploader for dataset
    file = st.file_uploader("Upload dataset CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)

        # Create start and end positions for the answers
        df['start_positions'] = df.apply(lambda row: find_answer_positions(row['Question'], row['Answer'])[0], axis=1)
        df['end_positions'] = df.apply(lambda row: find_answer_positions(row['Question'], row['Answer'])[1], axis=1)

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        st.write("Training Set Size:", len(train_df))
        st.write("Validation Set Size:", len(val_df))

        # Tokenize both datasets
        train_dataset = Dataset.from_pandas(train_df[['Question', 'Answer', 'start_positions', 'end_positions']])
        val_dataset = Dataset.from_pandas(val_df[['Question', 'Answer', 'start_positions', 'end_positions']])
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)

        # Remove unnecessary columns and set format for PyTorch
        tokenized_train = tokenized_train.remove_columns(['Question', 'Answer'])
        tokenized_val = tokenized_val.remove_columns(['Question', 'Answer'])
        
        tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
        tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
        
        model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val  
        )

        trainer.train()

        # For inference
        question = st.text_input("Ask your question")
        if st.button("Get Answer"):
            # Use the question as context
            inputs = tokenizer(question, question, padding='max_length', truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)

            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            start_index = torch.argmax(start_logits)
            end_index = torch.argmax(end_logits)

            answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]
            answer = tokenizer.decode(answer_tokens)

            st.write(f"Predicted Answer: {answer}")

if __name__ == '__main__':
    main()
