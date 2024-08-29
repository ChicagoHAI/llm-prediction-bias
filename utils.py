"""
Given an Intervenable object with BDAS intervention, 
return its rotation matrix and boundary masks
"""
def get_bdas_params(intervenable):
    key = list(intervenable.interventions.keys())[0]
    intervention = intervenable.interventions[key][0]
    rotate_layer = intervention.rotate_layer
    Q = rotate_layer.parametrizations.weight.original
    
    intervention_boundaries = intervention.intervention_boundaries
    intervention_boundaries = torch.clamp(intervention_boundaries, 1e-3, 1)
    
    boundary_mask = sigmoid_boundary(
        intervention.intervention_population, 
        0.,
        intervention_boundaries[0] * 4096,
        intervention.temperature
    )
    
    return intervention, Q, boundary_mask


def llm_predict(model, tokenizer, input_batch, 
                generate=False, gen_length=None):
    input_ids = tokenizer(input_batch, 
                          return_tensors="pt", 
                          padding=True).to(device)
    input_len = input_ids['input_ids'].shape[1]

    with torch.no_grad():
        if generate:
            output_ids = llama.generate(**input_ids, 
                                        max_length=input_len+gen_length)
            output_preds = tokenizer.batch_decode(output_ids[:, input_len:], 
                                                  skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=False)
        else:
            output_batch = model(**input_ids)
            output_ids = output_batch['logits'][:, -1, :].argmax(dim=-1)
            output_preds = tokenizer.batch_decode(output_ids, 
            skip_special_tokens=True)
            
    return output_preds