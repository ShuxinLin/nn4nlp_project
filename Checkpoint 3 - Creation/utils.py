import os
import torch


def check_and_move_to_cuda(*x):
  if torch.cuda.is_available():
    for i in x:
      i.cuda()


def get_latest_checkpoint(ckpt_dir):
  if not os.path.exists(ckpt_dir):
    return None

  all_model_ckpts = [name for name in os.listdir(ckpt_dir) if 'model' in name]
  all_metadata_ckpts = [name for name in os.listdir(ckpt_dir) if 'meta' in name]

  if len(all_model_ckpts) == 0:
    return None

  # latest versions should appear first
  all_model_ckpts = sorted(all_model_ckpts,
                           key=lambda name: int(name.split('_')[-1]),
                           reverse=True)
  all_metadata_ckpts = sorted(all_metadata_ckpts,
                              key=lambda name: int(name.split('_')[-1]),
                              reverse=True)

  # for restoring the latest
  model_latest_path = \
    os.path.abspath(os.path.join(ckpt_dir, all_model_ckpts[0]))
  metadata_latest_path = \
    os.path.abspath(os.path.join(ckpt_dir, all_metadata_ckpts[0]))

  return {
    'model': [all_model_ckpts, model_latest_path],
    'metadata': [all_metadata_ckpts, metadata_latest_path]
  }


def save_model(ckpt_dir,
               model,
               global_step,
               optimizer,
               desc='ner',
               n_versions=3):
  """
  Save model and relevant metadata 
  
  Args:
    ckpt_dir: 
    model:
    global_step: part of metadata 
    optimizer: part of metadata
    desc: name of model (to be used in the filename saved) 
    n_versions: max number of oldest versions retained (circular order)  
    
  """
  if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)

  # remove old checkpoint versions
  remove_old_ckpt_versions(ckpt_dir, n_versions)

  # save a new version of model
  model_save_path = \
    os.path.join(ckpt_dir, 'model_' + '_'.join([desc, str(global_step)]))
  torch.save(model, model_save_path)

  # and save metadata as well
  metadata = {'global_step': global_step,
              'optimizer': optimizer}
  metadata_save_path \
    = os.path.join(ckpt_dir, 'meta_' + '_'.join([desc, str(global_step)]))
  torch.save(metadata, metadata_save_path)

  print("\n\tModel saved to: {}\n".format(model_save_path))

  # return model_save_path, metadata_save_path


def remove_old_ckpt_versions(ckpt_dir, n_versions):
  """
  Clean up old versions before saving any new one, for both model and metadata  
  
  Args:
    ckpt_dir: 
    n_versions: max number of oldest versions to retain at any time  

  """
  ckpt_info = get_latest_checkpoint(ckpt_dir)

  if ckpt_info is None:
    return

  # get essential info first
  all_model_ckpts = ckpt_info['model'][0]
  all_metadata_ckpts = ckpt_info['metadata'][0]

  # cleaning
  if all_model_ckpts is not None:
    if len(all_model_ckpts) >= n_versions:

      # clean up old model versions
      model_paths_to_remove = all_model_ckpts[(n_versions - 1):]
      for path in model_paths_to_remove:
        os.remove(os.path.join(ckpt_dir, path))

      # and then old metadata versions
      metadata_paths_to_remove = all_metadata_ckpts[(n_versions - 1):]
      for path in metadata_paths_to_remove:
        os.remove(os.path.join(ckpt_dir, path))


def load_latest_model(ckpt_dir):
  ckpt_info = get_latest_checkpoint(ckpt_dir)

  if ckpt_info is None:
    print('\n' + '*' * 150 + '\n')
    print("\tCANNOT RESTORE FROM CHECKPOINT - TRAIN FROM SCRATCH... ")
    print('\n' + '*' * 150 + '\n')
    return None, None

  latest_model_path = ckpt_info['model'][1]
  latest_metadata_path = ckpt_info['metadata'][1]

  if torch.cuda.is_available():
    model = torch.load(latest_model_path)
    metadata = torch.load(latest_metadata_path)
  else:
    model = torch.load(latest_model_path,
                       map_location=lambda storage, loc: storage)
    metadata = torch.load(latest_metadata_path,
                          map_location=lambda storage, loc: storage)

  # to avoid the (annoying) contiguous warning in Pytorch
  model.flatten_parameters()

  print('\n' + '*' * 150 + '\n')
  print("*\tModel RESTORED from {}".format(latest_model_path))
  print('\n' + '*' * 150 + '\n')

  return model, metadata


def clip_grad_norm(parameters, max_norm, norm_type=2, writer=None, step=0):
    r""" FROM PYTORCH LIBRARY 
    
    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    if writer:
      writer.add_scalar('grad_norm', total_norm, global_step=step)

    return total_norm


if __name__ == "__main__":
  ckpt_info = get_latest_checkpoint("checkpoint")
  model_info = ckpt_info['model']
  meta_info = ckpt_info['metadata']

  print(model_info)
  print(meta_info)

  model, metadata = load_latest_model("checkpoint")
  print("Global step: {}".format(metadata['global_step']))
  import pdb; pdb.set_trace()