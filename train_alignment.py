# Standard library
import argparse
import os

# Third-party libraries
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import pyvene as pv

# Local libraries
from utils import save_alignment

"""
Calculate cross entropy between logits and 
a single target label (can be batched)
"""
def calculate_loss(intervenable, logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, intervenable.model_config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    boundary_loss = 0
    for k, v in intervenable.interventions.items():
        if isinstance(v[0], pv.BoundlessRotatedSpaceIntervention):
            boundary_loss = 1.0 * v[0].intervention_boundaries.sum()
    loss += boundary_loss

    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", help="""Path the the directory containing
                        the dataset files.""")
    parser.add_argument("--model_name", type=str, 
                        default='sharpbai/alpaca-7b-merged', 
                        help='Name or path of the model')
    parser.add_argument("--intervention_type", choices=["bdas", "das"], default="bdas")
    parser.add_argument("--interchange_dim", type=int)

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
    parser.add_argument("--train_end", action='store_true', 
                        help="Whether to train on tokens near the prompt's end."
                        )

    parser.add_argument("--n_train", type=int, default=-1)
    parser.add_argument("--n_dev", type=int, default=-1)
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
    vene_type = args.intervention_type
    interchange_dim = args.interchange_dim

    h_start = args.horizontal_start
    h_end = args.horizontal_end
    h_step = args.horizontal_step
    num_extra_steps = args.extra_steps
    train_end = args.train_end

    v_start = args.vertical_start
    v_end = args.vertical_end
    v_step = args.vertical_step

    n_train = args.n_train
    n_dev = args.n_dev
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    save_alignments = args.save_alignments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    })

    if n_train > 0:
        indices = np.random.choice(len(ds['train']), size=n_train, replace=True)
        ds['train'] = ds['train'].shuffle(seed=42).select(indices)

    if n_dev > 0:
        indices = np.random.choice(len(ds['dev']), size=n_dev, replace=True)
        ds['dev'] = ds['dev'].shuffle(seed=42).select(indices)

    train_loader = DataLoader(ds['train'], batch_size=batch_size)
    dev_loader = DataLoader(ds['dev'], batch_size=batch_size)

    if v_end == -1:
        v_end = llama.config.num_hidden_layers

    extra_steps = num_extra_steps * h_step
    layers = list(range(v_start, v_end+1, v_step))
    positions = list(range(h_start-extra_steps, h_end+1, h_step))

    if train_end:
        positions += list(range(-1-extra_steps, 0, h_step))

    # we search over layers and token positions
    for layer in layers:
        for position in positions:
            args.save_name = f"layer_{layer}_token_{position}"

            print(args.save_name)

            if vene_type == "bdas":
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
            else:
                config = pv.IntervenableConfig([
                    {
                        "layer": layer,
                        "component": 'block_output',
                        "intervention_type": pv.RotatedSpaceIntervention,
                    }
                ])
                intervenable = pv.IntervenableModel(config, llama)
                key = list(intervenable.interventions.keys())[0]
                # breakpoint()
                intervenable.interventions[key][0].interchange_dim = torch.tensor(interchange_dim)

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
                        'params': v[0].intervention_boundaries, 'lr': 1e-3
                        # 'params': v[0].intervention_boundaries, 'lr': 5e-3
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
                    
                    base = base_tokens.input_ids
                    src = source_tokens.input_ids

                    if position < 0:
                        base_pos = base.shape[1] + position
                        src_pos = src.shape[1] + position
                    else:
                        base_pos = src_pos = position

                    print(f"Base position: {base_pos}\nSource position: {src_pos}")

                    print("Base tokens:")
                    print(tokenizer.batch_decode(base[:10, base_pos]))
                    print("Source tokens:")
                    print(tokenizer.batch_decode(src[:10, src_pos]))

                    _, counterfactual_outputs = intervenable(
                        base_tokens,
                        [source_tokens],
                        unit_locations={"sources->base": (src_pos, base_pos)},
                    )

                    logits = counterfactual_outputs.logits[:, -1]
                    loss = calculate_loss(intervenable, logits, example['src_label'].to(device))

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
                        
                        base = base_tokens.input_ids
                        src = source_tokens.input_ids

                        if position < 0:
                            base_pos = base.shape[1] + position
                            src_pos = src.shape[1] + position
                        else:
                            base_pos = src_pos = position
                    
                        _, counterfactual_outputs = intervenable(
                            base_tokens,
                            [source_tokens],
                            unit_locations={"sources->base": (src_pos, base_pos)},
                        )

                        logits = counterfactual_outputs.logits[:, -1]
                        preds = logits.argmax(dim=-1).detach().cpu().numpy()

                        all_preds.append(preds)
                        all_labels.append(example['src_label'])
                        
                    all_preds = np.concatenate(all_preds)
                    all_labels = np.concatenate(all_labels)
                    acc = accuracy_score(all_preds, all_labels)

                    writer.add_scalar('dev accuracy', acc, epoch)
                    print(acc)

            # saving the alignment
            if save_alignments:
                save_alignment(intervenable, args.models_save_path, args.save_name)

            os.makedirs(args.results_save_path, exist_ok=True)
            with open(os.path.join(args.results_save_path, args.save_name + ".txt"), 'w') as fw:
                fw.write(f"Final dev accuracy: {acc:.2f}")