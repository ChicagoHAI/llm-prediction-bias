import argparse
import os
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup, \
LlamaForCausalLM, LlamaTokenizer, LlamaConfig, \
AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

import sys
sys.path.append('../pyvene/')
import pyvene as pv # using local pyvene

from utils import save_alignment

"""
Calculate cross entropy between logits and 
a single target label (can be batched)
"""
def calculate_loss(logits, labels):
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_labels = labels.to(logits.device)
    loss = loss_fct(logits, shift_labels)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", help="""Path the the directory containing
                        the dataset files.""")
    parser.add_argument("--model_name", type=str, 
                        default='sharpbai/alpaca-7b-merged', 
                        help='Name or path of the model')

    # Training args
    parser.add_argument("--horizontal_start", type=int, default=0)
    parser.add_argument("--horizontal_end", type=int, default=50)
    parser.add_argument("--horizontal_step", type=int, default=1)

    parser.add_argument("--vertical_start", type=int, default=0)
    parser.add_argument("--vertical_end", type=int, default=-1)
    parser.add_argument("--vertical_step", type=int, default=1)

    parser.add_argument("--extra_steps", 
                        help="""The number of steps before {h_pos} to search.""", 
                        default=4, type=int)

    parser.add_argument("--num_epochs", help="""Number of training epochs.""",
                        default=1, type=int)
    parser.add_argument("--batch_size", help="""Training batch size.""",
                        default=32, type=int)

    parser.add_argument("--save_alignments", 
                        help="""Whether to save the resulting
                            alignment or not.""",
                        action='store_true')
    parser.add_argument("--models_save_path", 
                        help="""Path to save the resulting models.
                        Should end in a directory.""")
    parser.add_argument("--results_save_path", 
                        help="""Path to the directory to save
                        the dev accuracies.""")

    args = parser.parse_args()

    ds_path = args.dataset_path
    model_name = args.model_name

    """
    Alpaca:
    - admissions: 16 or 9 (p_var is 29, prod_var is 61)
    - admissions_race_offset: 62
    - hire_dec: 18
    - hire_dec_eval: 18
    - hire_dec_names: 17
    Mistral:
    - admissions: 43
    - hire_dec: 40
    - hire_dec_eval: 18
    - hire_dec_names: 17
    Gemma:
    - admissions: 14
    - hire_dec: 15
    - hire-dec-eval: 15
    - hire-dec-names: 14
    """

    h_start = args.horizontal_start
    h_end = args.horizontal_end
    h_step = args.horizontal_step
    num_extra_steps = args.extra_steps

    v_start = args.vertical_start
    v_end = args.vertical_end
    v_step = args.vertical_step

    num_epochs = args.num_epochs
    batch_size = args.batch_size

    save_alignments = args.save_alignments
    device = 'cuda'

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    llama = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,  # save memory
    )
    _ = llama.to(device)
    _ = llama.eval()

    ds = load_dataset('csv', data_files={
        'train': os.path.join(ds_path, 'train.csv'),
        'dev': os.path.join(ds_path, 'dev.csv'),
        'test': os.path.join(ds_path, 'test.csv'),
    })

    train_loader = DataLoader(ds['train'].shuffle(seed=42).select(range(500)), batch_size=batch_size)
    dev_loader = DataLoader(ds['dev'], batch_size=batch_size)
    test_loader = DataLoader(ds['test'], batch_size=batch_size)

    if v_end == -1:
        v_end = llama.config.num_hidden_layers

    # max_seq_len = len(tokenizer(ds['train'][0]['base']).input_ids)
    token_ids = tokenizer(ds['train']['base'][:50], 
                        padding=True, 
                        return_tensors="pt").input_ids
    max_seq_len = token_ids.shape[1]

    extra_steps = num_extra_steps * h_step

    layers = list(range(v_start, v_end+1, v_step))

    positions = list(range(h_start-extra_steps, h_end+1, h_step)) \
    + list(range((max_seq_len-1)-extra_steps, max_seq_len, h_step))

    # we search over layers and token positions
    for layer in layers:
        for position in positions:
            args.save_name = f"layer_{layer}_token_{position}"

            print(args.save_name)

            config = pv.IntervenableConfig([
                {
                    "layer": layer,
                    "component": 'block_output',
                    "intervention_type": pv.BoundlessRotatedSpaceIntervention,
                }
            ])
            intervenable = pv.IntervenableModel(config, llama)
            intervenable.set_device(device)
            intervenable.disable_model_gradients()

            # set up optimizer
            total_steps = num_epochs * len(ds['train'])
            optimizer_params = []
            for k, v in intervenable.interventions.items():
                try:
                    optimizer_params.append({
                        "params": v[0].rotate_layer.parameters()
                    })
                    optimizer_params.append({
                        'params': v[0].intervention_boundaries, 'lr': 1e-2
                    })
                except:
                    pass
            optimizer = torch.optim.Adam(optimizer_params, lr=1e-4)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            ) 

            # setting up tensorboard for loss visualization
            tsboard_path = os.path.join(
                f'./tensorboard/{os.path.basename(model_name)}', 
                args.save_name
            )
            os.makedirs(tsboard_path, exist_ok=True)
            writer = SummaryWriter(tsboard_path)

            curr_step = 0
            for epoch in range(num_epochs):
                epoch_iterator = tqdm(
                    train_loader, desc=f"Epoch: {epoch}", position=0, leave=True
                )
                # training loop
                for example in epoch_iterator:
                    base_tokens = tokenizer(example['base'], 
                                            return_tensors='pt', 
                                            padding=True).to(device)
                    source_tokens = tokenizer(example['source'], 
                                            return_tensors='pt', 
                                            padding=True).to(device)

                    _, counterfactual_outputs = intervenable(
                        base_tokens,
                        [source_tokens],
                        {"sources->base": position},
                    )

                    logits = counterfactual_outputs.logits[:, -1]
                    loss = calculate_loss(logits, example['src_label'].to(device))
                    epoch_iterator.set_postfix({"loss": f"{loss.item():.3f}"})
                    
                    writer.add_scalar('training loss', loss, curr_step)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    curr_step += 1

                # eval
                with torch.no_grad():
                    iterator = tqdm(dev_loader)
                    all_preds = []
                    all_labels = []
                    
                    for example in iterator:
                        base_tokens = tokenizer(example['base'], 
                                                return_tensors='pt', 
                                                padding=True).to(device)
                        source_tokens = tokenizer(example['source'], 
                                                return_tensors='pt', 
                                                padding=True).to(device)
                    
                        _, counterfactual_outputs = intervenable(
                            base_tokens,
                            [source_tokens],
                            {"sources->base": position},
                        )

                        logits = counterfactual_outputs.logits[:, -1]
                        preds = logits.argmax(dim=-1).detach().cpu().numpy()

                        # print(preds)

                        all_preds.append(preds)
                        all_labels.append(example['src_label'])
                        
                    all_preds = np.concatenate(all_preds)
                    all_labels = np.concatenate(all_labels)
                    acc = accuracy_score(all_preds, all_labels)

                    # print(all_preds[:20])

                    writer.add_scalar('dev accuracy', acc, epoch)

            # saving the alignment
            if save_alignments:
                save_alignment(intervenable, args.models_save_path, args.save_name)

            os.makedirs(args.results_save_path, exist_ok=True)
            with open(os.path.join(args.results_save_path, args.save_name + ".txt"), 'w') as fw:
                fw.write(f"Final dev accuracy: {acc:.2f}")