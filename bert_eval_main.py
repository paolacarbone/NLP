import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Sequence
from transformers import BertTokenizerFast, BertForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, TrainerCallback
import evaluate
from functools import partial
import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str,
                    help="""Choice of BERT model.
                    options:
                    materialsbert: MaterialsBERT by Ramprasaad Group
                    matbert_cased: MatBERT by Ceder group
                    matbert_uncased: MatBERT by Ceder group
                    scibert_cased: SciBERT by AllenAI
                    scibert_uncased: SciBERT by AllenAI
                    bert_uncased: BERT base uncased
                    bert_cased: BERT base cased
                    matscibert: MatSciBERt by T. Gupta et al""")
parser.add_argument("-o", "--output", type=str,
                    help="Output directory to save the evaluation results")
parser.add_argument("-s", "--model_save", type=str,
                    help="Output directory to save the model")
parser.add_argument("-c", "--corpus", type=str,
                    help="""Choice of corpus to use for training and testing
                    options:
                    polyie: PolymerIE dataset
                    polymerabstracts
                    matscholar: MatScholar dataset
                    pcmsp: PC-MSP dataset""")
args = parser.parse_args()

class load_corpus():
    """Load a corpus for training and testing"""
    def __init__(self, corpus):
        self.corpus = corpus

        # Load and preprocess the dataset
        ner_data, id2label, label2id, label_list = self.choose_corpus()
        ner_data = self.preprocessing(ner_data)
        ner_data = self.get_labels(ner_data, label_list)
        ner_data = ner_data.map(self.truncate_after_first_full_stop)

        # Store results
        self.ner_data = ner_data
        self.id2label = id2label
        self.label2id = label2id
        self.label_list = label_list

        return


    def choose_corpus(self):
        """
        Load a specified dataset and return the processed HuggingFace DatasetDict
        along with label mappings.
        """
        def split_long_sequences(batch, max_length=200):
            """Split long sequences at sentence boundaries if possible."""
            def check_string(s):
                """Check if the token is a full stop at the end of a sentence."""
                return s == "." or (s.endswith('.') and any(not c.isdigit() for c in s[:-1]))

            new_tokens = []
            new_labels = []
            temp_tokens = []
            temp_labels = []

            # Loop through the tokens and labels in the batch
            for tokens, labels in zip(batch['words'], batch['ner']):
                i = 0
                for token, label in zip(tokens, labels):
                    # Add the token to the temporary list
                    temp_tokens.append(token)
                    temp_labels.append(label)
                    i += 1
                    
                    # Once the temporary list reaches the max length, check for a full stop
                    if i >= max_length and check_string(token):
                        new_tokens.append(temp_tokens)
                        new_labels.append(temp_labels)
                        temp_tokens = []
                        temp_labels = []
                        i = 0
                
                # Append any remaining tokens that were not added
                if len(temp_tokens) > 0:
                    new_tokens.append(temp_tokens)
                    new_labels.append(temp_labels)

            return {
                'words': new_tokens,
                'ner': new_labels
            }

        def load_pcmsp():
            def nerify(datas):
                dataset = []
                for i, data in enumerate(datas):
                    ner = ["O" for _ in range(len(data["tokens"]))]
                    ner_types = [data["vertexSet"][a]["kbID"] for a in range(len(datas[i]["vertexSet"]))]
                    ner_pos = [data["vertexSet"][a]["tokenpositions"] for a in range(len(datas[i]["vertexSet"]))]
                    for j, nerd in enumerate(ner_types):
                        prefix = "B-"
                        for k in ner_pos[j]:
                            ner[k] = prefix+nerd
                            prefix = "I-"
                    dataset.append({
                        "words": data["tokens"],
                        "ner": ner
                    })
                dataset = Dataset.from_dict({
                    "words": [example['words'] for example in dataset],
                    "ner": [example['ner'] for example in dataset]
                })
                return dataset
            with open("pcmsp_train.json", "r") as file:
                train_dataset = nerify(json.load(file))
            with open("pcmsp_test.json", "r") as file:
                test_dataset = nerify(json.load(file))
            dataset_dict = DatasetDict({
                "train": train_dataset,
                "validation": test_dataset
            })
            return dataset_dict

        def load_polymer_abstracts():
            df=pd.read_json("polymerabstracts_train.json", lines=True)
            train_dataset = Dataset.from_pandas(df)#.select(range(20))

            df=pd.read_json("polymerabstracts_test.json", lines=True)
            test_dataset = Dataset.from_pandas(df)#.select(range(20))

            dataset_dict = DatasetDict({
                "train": train_dataset,
                "validation": test_dataset
            })
            return dataset_dict

        def load_matscholar():
            with open("matscholar.json", "r") as file:
                data_train = json.load(file)
            useful_data = [data_train["data"][i][1] for i in range(len(data_train["data"]))]
            
            texts = []
            ner = []

            for datum in useful_data:
                words = list(datum.keys())
                labels = list(datum.values())
                texts.append(words)
                ner.append(labels)

            matscholar = Dataset.from_dict({
                        "words": texts,
                        "ner": ner
                    })#.select(range(50))
            split_dataset = matscholar.train_test_split(test_size=0.2)

        # Extract training and validation sets
            train_dataset = split_dataset["train"]
            val_dataset = split_dataset["test"]

            return DatasetDict({
                "train": train_dataset,
                "validation": val_dataset
            })

        def load_polyie():
            def flattened_data(data):
                flattened_text = [item for sublist in data["text"] for item in sublist]
                flattened_label = [item for sublist in data["label"] for item in sublist]
                return Dataset.from_dict({
                    "text": flattened_text,
                    "label": flattened_label
                })

            with open("polyie_train.txt", "r") as file:
                data_train = flattened_data(json.load(file))  # Safe method
            with open("polyie_validation.txt", "r") as file:
                data_validate = flattened_data(json.load(file))  # Safe method

            dataset = DatasetDict({
                "train": data_train,
                "validation": data_validate
            })
            dataset = dataset.rename_columns({'text':'words','label': 'ner'})
            dataset = dataset.map(split_long_sequences, batched=True)
            return dataset

        corpus = self.corpus.lower()

        if corpus == "polymerabstracts":
            label_list = ["O", "I-POLYMER", "I-PROP_VALUE", "I-PROP_NAME", "I-MONOMER", 
                "I-ORGANIC", "I-INORGANIC", "I-MATERIAL_AMOUNT", "I-POLYMER_FAMILY"]
            
            id2label = {i: label for i, label in enumerate(label_list)}
            label2id = {v: k for k, v in id2label.items()}

            dataset = load_polymer_abstracts()
            
        elif corpus == "polyie":
            label_list = ['I-PV', 'B-PN', 'B-CN', 'B-ES', 'I-PN', 'B-PV', 'B-Condition', 'I-ES', 'I-Condition', 'I-CN', 'O']
            
            id2label = {i: label for i, label in enumerate(label_list)}
            label2id = {v: k for k, v in id2label.items()}

            dataset = load_polyie()

        elif corpus == "pcmsp":
            label_list = ['B-Property-rate', 'I-Brand', 'B-Value', 'I-Value', 'O', 'B-Property-time', 'B-Property-temperature', 
                        'I-Material-intermedium', 'B-Material-intermedium', 'I-Material-recipe', 'I-Property-pressure', 'I-Material-target', 
                        'I-Property-rate', 'I-Property-temperature', 'B-Material-others', 'I-Property-time', 'B-Device', 'B-Material-recipe', 'B-Brand', 
                        'I-Device', 'B-Descriptor', 'B-Property-pressure', 'I-Operation', 'B-Operation', 'I-Material-others', 'I-Descriptor', 'B-Material-target']
            
            id2label = {i: label for i, label in enumerate(label_list)}
            label2id = {v: k for k, v in id2label.items()}

            dataset = load_pcmsp()
        
        elif corpus == "matscholar":
            label_list = ['B-SMT', 'B-DSC', 'B-MAT', 'I-DSC', 'I-PRO', 'O', 'B-CMT', 'I-MAT', 'I-APL', 'I-SMT', 'B-SPL', 'B-APL', 'I-SPL', 'I-CMT', 'B-PRO']
            
            id2label = {i: label for i, label in enumerate(label_list)}
            label2id = {v: k for k, v in id2label.items()}

            dataset = load_matscholar()

        else:
            raise ValueError("Invalid corpus name. Please choose from: polymerabstracts, polyie, pcmsp, matscholar")

        return dataset, id2label, label2id, label_list
    

    def preprocessing(self, dataset_dict):
        """Ensure all tags start with B-/I-/O and rename columns for consistency."""
        def add_prefix(example):
            example["ner"] = [
                tag if tag.startswith(("B-", "I-")) or tag == "O" else f"I-{tag}"
                for tag in example["ner"]
            ]
            return example

        dataset_dict = dataset_dict.map(lambda example, idx: {"text_id": idx}, with_indices=True)
        dataset_dict = dataset_dict.map(add_prefix)
        dataset_dict = dataset_dict.rename_columns({'words': 'text', 'ner': 'ner_tags'})
        return dataset_dict

    def get_labels(self, dataset_dict, label_list):
        """Map string NER labels to integer class IDs using ClassLabel."""
        def convert_labels(example):
            # Ensure labels is a list of strings and each label is converted to integer
            example["ner_tags"] = [classmap.str2int(tag) for tag in example["ner_tags"]]
            return example
        
        classmap = ClassLabel(num_classes=len(label_list), names=label_list)
        dataset_dict = dataset_dict.map(convert_labels)
        dataset_dict = dataset_dict.cast_column("ner_tags", Sequence(classmap))
        return dataset_dict
    
    
    def truncate_after_first_full_stop(self, example):
        """Truncate sequences after the first full stop."""
        def check_string(s):
            """Check if the token is a full stop at the end of a sentence."""
            return s == "." or (s.endswith('.') and any(not c.isdigit() for c in s[:-1]))

        max_length = 400
        truncated_tokens = []
        truncated_labels = []
        temp_tokens = []
        temp_labels = []

        for tokens, labels in zip(example["text"], example["ner_tags"]):
            if len(truncated_tokens) <= max_length:
                truncated_tokens.append(tokens)
                truncated_labels.append(labels)
                continue

            # Find first full stop after index 450
            # for i in range(max_length, len(example["text"])):
            if check_string(tokens):
                temp_tokens.append(tokens)  # Include the full stop
                temp_labels.append(labels)
                truncated_tokens.extend(temp_tokens)
                truncated_labels.extend(temp_labels)
                break
            else:
                    # If no full stop found, keep the full sequence
                temp_tokens.append(tokens)
                temp_labels.append(labels)
        example["text"] = truncated_tokens
        example["ner_tags"] = truncated_labels
        return example


class load_model():
    """Load a model for training and testing"""
    def __init__(self, model, id2label, label2id):
        self.model_choice = model
        self.id2label = id2label
        self.label2id = label2id
        self.tokenizer, self.model = self.choose_model()
        return

    
    def choose_model(self):
        model = self.model_choice.lower()

        if model == "matbert_uncased":
            model_checkpoint = "matbert-base-uncased"
            tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, model_max_len=512)

            model = BertForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        if model == "matbert_cased":
            model_checkpoint = "matbert-base-cased"
            tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, model_max_len=512)

            model = BertForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        elif model == "materialsbert":
            model_checkpoint = "pranav-s/MaterialsBERT"
            tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, model_max_len=512)

            model = BertForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        elif model == "scibert_uncased":
            model_checkpoint = "allenai/scibert_scivocab_uncased"
            tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, model_max_len=512)

            model = BertForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        elif model == "scibert_cased":
            model_checkpoint = "allenai/scibert_scivocab_cased"
            tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, model_max_len=512)

            model = BertForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        elif model == "matscibert":
            model_checkpoint = "m3rg-iitd/matscibert"
            tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, model_max_len=512)

            model = BertForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        elif model == "bert_cased":
            model_checkpoint = "google-bert/bert-large-uncased"
            tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, model_max_len=512)

            model = BertForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        elif model == "bert_uncased":
            model_checkpoint = "google-bert/bert-large-cased"
            tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, model_max_len=512)

            model = BertForTokenClassification.from_pretrained(
                model_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        else:
            raise ValueError("Invalid model name. Please choose from: matbert_uncased, matbert_cased, materialsbert, scibert_uncased, scibert_cased, matscibert, bert_cased, bert_uncased")

        return tokenizer, model
    
class evaluate_corpus_model():
    """Evaluate the model on the test set"""
    def __init__(self, model, corpus, model_save, eval_output):
        self.batch_size = 16

        # Load data and model
        self.corpus = load_corpus(corpus)
        self.model = load_model(model, self.corpus.id2label, self.corpus.label2id)

        # Tokenize text and align labels
        tokenized_datasets = self.corpus.ner_data.map(
            lambda x: self.tokenize_and_align_labels(x, self.model.tokenizer), 
            batched=True,
            remove_columns=self.corpus.ner_data["train"].column_names,
        )

        # Filter sequences that are too long
        self.tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) < 512)

        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.model.tokenizer, return_tensors="pt")

        # Setup and train
        self.trainer = self.make_trainer()
        self.trainer.train()

        # Plot and save loss/metrics
        self.save_loss_to_json(self.trainer.state.log_history, "eval_results/"+self.cheeky_png(eval_output))

        # Save model and tokenizer
        if model_save:
            self.trainer.save_model(model_save)
            self.model.tokenizer.save_pretrained(model_save)

        # Save evaluation results to JSON
        eval_results = self.trainer.evaluate()
        with open("eval_results/"+eval_output, "w") as f:
            json.dump(eval_results, f, indent=4)
        return
        
    
    def tokenize_and_align_labels(self, examples, tokenizer):
        """Align original word-level NER tags with WordPiece tokens."""
        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    # Special token
                    new_labels.append(-100)
                else:
                    # Same word as previous token
                    label = labels[word_id]
                    # If the label is B- we change it to I-
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)

            return new_labels
        
        tokenized_inputs = tokenizer(
            examples["text"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    def compute_metrics(self, p, label_list):
        seqeval = evaluate.load("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return results
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    
    def set_training_args(self):
        training_args = TrainingArguments(
        learning_rate=5e-5,
        per_device_train_batch_size=self.batch_size,
        per_device_eval_batch_size=self.batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy='epoch',
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model = "eval_overall_f1",
        )
        return training_args
    
    def make_trainer(self):
        trainer = Trainer(
        model=self.model.model,
        args=self.set_training_args(),
        train_dataset=self.tokenized_datasets["train"],
        eval_dataset=self.tokenized_datasets["validation"],
        processing_class=self.model.tokenizer,
        data_collator=self.data_collator,
        compute_metrics=partial(self.compute_metrics, label_list=self.corpus.label_list),
    )
        return trainer
    
    
    def save_loss_to_json(self, losses, filename):
        train_loss = []
        eval_loss = []
        overall_f1 = []
        for elem in losses:
            if 'loss' in elem.keys():
                train_loss.append(elem['loss'])
            if 'eval_loss' in elem.keys():
                eval_loss.append(elem['eval_loss'])
            if 'eval_overall_f1' in elem.keys():
                overall_f1.append(elem['eval_overall_f1'])
                
        epoch = list(range(1, len(train_loss) + 1))
        
        fig, ax1 = plt.subplots()
        ax1.plot(epoch, train_loss, label="Train Loss", color='blue')
        ax1.plot(epoch, eval_loss, label="Eval Loss", color='green')
        ax2 = ax1.twinx()
        ax2.plot(epoch, overall_f1, label="f1", color='orange')
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.savefig(filename, format='png')
        return
    
    def cheeky_png(self, s):
        before_dot, dot, _ = s.rpartition(".")
        return before_dot + dot + "png" if dot else s + ".png"
    

eval = evaluate_corpus_model(args.model, args.corpus, args.model_save, args.output)
print("Evaluation complete. Results saved to eval_results/"+args.output)