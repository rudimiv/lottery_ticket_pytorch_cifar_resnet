import torch
import torch.nn.utils.prune as prune


'''
We globally prune convolutions at a rate of 20% per iteration. 
We do not prune the 2560 parameters used to downsample residual connections or the 
640 parameters in the fully-connected output layer, 
as they comprise such a small portion of the overall network.
'''

def print_statisitcs_by_masks(model):
    total_pruned = 0
    total_params = 0

    for param_name, param in model.named_buffers():
        # print(param_name)
        if 'weight_mask' in param_name and 'conv' in param_name:
            total_pruned += torch.sum(param == 0).item()
            total_params += param.nelement()

            print(f'{param_name:30}: {torch.sum(param == 0).item()} / {param.nelement()} =' \
                  f'{torch.sum(param == 0).item() / param.nelement():.4f}')

    if total_params:
        print(f'total: {total_pruned} / {total_params} = {total_pruned / total_params}')


# it takes into account previous masks
def prune_model(model, percent=None):
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))

    if percent:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=percent
        )
    else:
        print('Identity pruning...')
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.Identity
        )

    print_statisitcs_by_masks(model)