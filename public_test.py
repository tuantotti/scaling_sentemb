import re
import sys
import io, os
import torch
import numpy as np
import logging
import tqdm
import fcntl
import time
import argparse
from prettytable import PrettyTable
import transformers
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.metrics.pairwise import paired_euclidean_distances
from typing import List, Optional
import matplotlib.pyplot as plt
import itertools
import pandas as pd
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                # Acquire an exclusive lock (non-blocking)
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Perform your write operations here
                file.write(content + '\n')
                file.flush()

            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                # Release the lock
                fcntl.flock(file, fcntl.LOCK_UN)
                break

def convert_triple_to_double_data(
    test_df, label_predict, euclidean_distances
):
    """
    Convert triple data to double data.

    This function reads data from a given test data path and converts it from triple format
    (query, positive_similar, negative_similar) to double format (query, doc_related, true_label).

    Args:
        test_df (DataFrame)
        preprocess_data (bool): Whether to preprocess the data or not. Default is False.

    Returns:
        pd.DataFrame: The converted double data as a pandas DataFrame with columns 'query', 'doc_related', and 'true_label'.
    """
    test = test_df.values.tolist()
    query = []
    doc_related = []
    true_label = []
    for record in test:
        if record[1]:
            query.append(record[0])
            doc_related.append(record[1])
            true_label.append(float(1))
        if record[2]:
            query.append(record[0])
            doc_related.append(record[2])
            true_label.append(float(0))

    data_df = pd.DataFrame.from_dict(
        {
            "query": query,
            "doc_related": doc_related,
            "true_label": true_label,
            "pred_labels": label_predict,
            "euclidean_distances": euclidean_distances,
        }
    )
    return data_df

def plot_confusion_matrix(
    cm,
    target_names: Optional[List] = None,
    title: str = "Confusion matrix",
    cmap=None,
    normalize: bool = True,
    save_name: str = "confusion_matrix",
    save_dir: str = None,
) -> None:
    """Function to plot the confusion matrix.
    Args:
        cm: The confusion matrix from `sklearn.metrics.confusion_matrix`.
        target_names (Optional[List]): The given classification classes such as
            [0, 1, 2] or the class name, for example ["clean", "hate"].
        title (str, default="Confusion matrix"): The text to display at the top of the matrix.
        cmap: The gradient of the values displayed from matplotlib.pyplot.cm
            see http://matplotlib.org/examples/color/colormaps_reference.html
            plt.get_cmap('jet') or plt.cm.Blues.
        normalize (bool, default=True): If False, plot the raw numbers.
            If True, plot the proportions.
        save_name (str): The name of the image file.
        save_dir (str): The directory to save the confusion matrix.
    Returns:
        `None`
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1.0 - accuracy
    if cmap is None:
        cmap = plt.get_cmap("Blues")

    width = int(10 / 4 * len(target_names))
    height = int(10 / 4 * len(target_names))
    plt.figure(figsize=(width, height))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.2f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(f"Predicted label\naccuracy={accuracy:0.4f}; " f"misclass={misclass:0.4f}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    try:
        save_path = os.path.join(save_dir, f"{save_name}{'_normalize' if normalize else ''}.png")
        plt.savefig(save_path, format="png", bbox_inches="tight")
    except IOError:
        print(f"Could not save file in directory: {save_dir}")
        
def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows)-1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i+1][0]) / 2

        return max_acc, best_threshold
    
def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows)-1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold

def save_json(filename: str, obj: dict):
    """
    Save a Python object as JSON to a file.

    Args:
        filename (str): The name of the file to save the JSON data to.
        obj (dict): The Python object to be saved as JSON.

    Returns:
        None
    """
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(obj, json_file, ensure_ascii=False, indent=4, separators=(",", ": "))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na'],
                        default='sts',
                        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument('--load_kbit', type=int,
                        choices=[4,8,16],
                        default=8,
                        help="Load model in kbit")

    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--lora_weight', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--tensor_parallel', action='store_true')
    parser.add_argument('--dense_words', type=str, default=None)
    parser.add_argument('--icl_examples_file', type=str, default=None)
    parser.add_argument('--dense_words_idx', type=int, default=-1)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--public_test_folder', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)


    args = parser.parse_args()

    auth_config = {}
    print(args.model_name_or_path)
    if "PhoGPT" in args.model_name_or_path:
        auth_config = {
            "trust_remote_code": True,
            "use_auth_token": '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'
        }

    if args.tensor_parallel:
        import tensor_parallel as tp
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    if args.load_kbit == 4:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.float16,
            device_map={"":0},
            **auth_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     output_hidden_states=True,
                                                     trust_remote_code=True,
                                                     load_in_8bit=args.load_kbit == 8,)

    if args.lora_weight is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            args.lora_weight,
            torch_dtype=torch.float16,
            device_map={'': 0},
        )
        if args.load_kbit == 4:
            from peft.tuners.lora import LoraLayer
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    #module = module.to(torch.bfloat16)
                    module = module.to(torch.float16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        #module = module.to(torch.bfloat16)
                        if 'opt' in args.model_name_or_path:
                            module = module.to(torch.float32)
                        else:
                            module = module.to(torch.float16)


    if 'llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = '</s>'
        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **auth_config)

    tokenizer.padding_side = "right"  # Allow batched inference

    if args.vocab_file:
      # Read existing vocabulary from the tokenizer
      existing_vocab = set(tokenizer.get_vocab().keys())
      # Read vocabulary from the vocab file
      with open(args.vocab_file, "r", encoding="utf-8") as f:
          additional_vocab = [word.strip() for word in f]
      new_vocab = [word for word in additional_vocab if word not in existing_vocab]
      if new_vocab:
          tokenizer.add_tokens(new_vocab)
      model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_embeddings(model, batch):
        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        # Get raw embeddings
        with torch.no_grad():
            hidden_states = model(return_dict=True, output_hidden_states=True, **batch).hidden_states
            if args.avg:
                last_layer = hidden_states[-1]
                #   attention_mask = batch['attention_mask'].unsqueeze(-1).expand(last_layer.shape)
                outputs = (last_layer).mean(1)
            else: # last token
                outputs = hidden_states[-1][:, -1, :]

            if outputs.dtype == torch.bfloat16:
                # bfloat16 not support for .numpy()
                outputs = outputs.float()

            return outputs.cpu()
    result_folder = f"public_test_results_{args.model_name_or_path.split()[-1]}"
    out_file_name = f"{args.model_name_or_path.split()[-1]}-public-test.csv"
    os.makedirs(result_folder, exist_ok=True)
    
    if args.public_test_folder:
        list_files = os.listdir(args.public_test_folder)
        for f in list_files:
            with open(args.public_test_folder+"/"+f) as file:
                data = json.load(file)
            
            sentences1 = []
            sentences2 = []
            all_labels = []
            subject_name = f.split("/")[-1].split(".")[0]
            embeddings1 = []
            embeddings2 = []

            for d in data:
                if d['pos_similar'] != "":
                    sentences1.append(d['query'])
                    sentences2.append(d['pos_similar'])
                    all_labels.append(1)

                if d['neg_similar'] != "":
                    sentences1.append(d['query'])
                    sentences2.append(d['neg_similar'])
                    all_labels.append(0)
                    
            num_batches = len(sentences1) // args.batch_size
            remainder = len(sentences1) % args.batch_size
            
            for i in range(num_batches):
                batch1 = tokenizer.batch_encode_plus(
                    sentences1[i * args.batch_size: (i + 1) * args.batch_size],
                    return_tensors='pt',
                    padding=True,
                    max_length=args.max_length,
                    truncation=True
                )

                batch2 = tokenizer.batch_encode_plus(
                    sentences2[i * args.batch_size: (i + 1) * args.batch_size],
                    return_tensors='pt',
                    padding=True,
                    max_length=args.max_length,
                    truncation=True
                )

                embeddings1.extend(get_embeddings(model, batch1).numpy())
                embeddings2.extend(get_embeddings(model, batch2).numpy())
            
            if remainder > 0:
                batch1 = tokenizer.batch_encode_plus(
                    sentences1[-remainder:],
                    return_tensors='pt',
                    padding=True,
                    max_length=args.max_length,
                    truncation=True
                )

                batch2 = tokenizer.batch_encode_plus(
                    sentences2[-remainder:],
                    return_tensors='pt',
                    padding=True,
                    max_length=args.max_length,
                    truncation=True
                )

                embeddings1.extend(get_embeddings(model, batch1).numpy())
                embeddings2.extend(get_embeddings(model, batch2).numpy())
                
            all_labels = np.asarray(all_labels)
            euclidean_distances = paired_euclidean_distances(np.array(embeddings1), np.array(embeddings2))
            output_scores = {}
            scores, reverse = euclidean_distances, False
            acc, acc_threshold = find_best_acc_and_threshold(scores, all_labels, reverse)
            f1, precision, recall, f1_threshold = find_best_f1_and_threshold(scores, all_labels, reverse)
            ap = average_precision_score(all_labels, scores * (1 if reverse else -1))

            output_scores["Euclidean-Distance"] = {
            "accuracy": acc,
            "accuracy_threshold": acc_threshold,
            "f1": f1,
            "f1_threshold": f1_threshold,
            "precision": precision,
            "recall": recall,
            "ap": ap,
            }

            label_predict = []
            for score in euclidean_distances:
                if score > output_scores["Euclidean-Distance"]["f1_threshold"]:
                    label_predict.append(float(0))
                else:
                    label_predict.append(float(1))

            cfm = confusion_matrix(y_true=all_labels, y_pred=label_predict, labels=[1.0, 0.0])

            # Plot the confusion matrix with the raw numbers
            plot_confusion_matrix(
                cm=cfm,
                target_names=["Pos_Similar", "Neg_Similar"],
                normalize=False,
                save_name=f'{subject_name}',
                save_dir=f"{result_folder}",
            )
            # Plot the confusion matrix with the proportions
            plot_confusion_matrix(
                cm=cfm,
                target_names=["Pos_Similar", "Neg_Similar"],
                normalize=True,
                save_name=f'{subject_name}-normalized',
                save_dir=f"{result_folder}",
            )

            data_df = pd.DataFrame.from_dict({
                "query": sentences1,
                "doc_related": sentences2,
                "true_label": all_labels,
                "pred_labels": label_predict,
                "euclidean_distances": euclidean_distances,
            })
            
            result_df = pd.DataFrame.from_dict(output_scores['Euclidean-Distance'], orient='index', columns=[subject_name]).transpose()

            print(f"subject: {subject_name}\n Result:{output_scores}\n")
            save_json(f"{result_folder}/{subject_name}.json", output_scores)
            data_df.to_csv(f"{result_folder}/{subject_name}_results_evaluation.csv")

            public_test_df = pd.concat([public_test_df, result_df])

        print(public_test_df)
        public_test_df.to_csv(result_folder+"/"+out_file_name)

if __name__ == "__main__":
    main()
