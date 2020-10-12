import numpy as np
import torch
from torch_scatter import scatter


def Global(x_0, u_0, entity_labels = None):
    r"""
    Estimate unkonwns based on avergaing over all nodes OR the entity types

    x_0 : torch.tensor of initial values with unkown zero-padded
    u_0 : torch.tensor of indicator of known values
    entity_labels : numpy.array(object)

    returns prediction as torch.tensor()
    """
    global_avg = x_0[u_0 == 1].mean().item()
    x = x_0.clone()

    if entity_labels is None:
      x[u_0 == 0] = global_avg
    else:
      for entity in np.unique(entity_labels):
        entity_mask = torch.tensor(entity_labels == entity, device=u_0.device)
        if any(entity_mask & (u_0 == 1)):
          global_entity_avg = x_0[entity_mask & (u_0 == 1)].mean().item()
        else:
          global_entity_avg = global_avg
        x[entity_mask & (u_0 == 0)] = global_entity_avg

    return x

def Local(edge_list, x_0, u_0, entity_labels = None):
    r"""
    Estimate unknowns based on local avergaing on all OR entity types

    edge_list: list of non-duplicate edges
    x_0 : torch.tensor of initial values with unkown zero-padded
    u_0 : torch.tensor of indicator of known values
    entity_labels : numpy.array(object)

    returns prediction as torch.tensor()
    """
    edge_array = np.concatenate(edge_list) # merge
    edge_array = np.concatenate((edge_array, np.flip(edge_array,1)))
    ent_avg = torch.ones_like(x_0)*x_0[u_0 == 1].mean().item()

    if not entity_labels is None:
      # Reduce edge list by removing cross entity edges
      egde2label = entity_labels[edge_array]
      edge_array = edge_array[np.where(egde2label[:,0] == egde2label[:,1])[0], :]
      for label in np.unique(entity_labels):
        avg = x_0[u_0.cpu().data.numpy() & (entity_labels == label)].mean().item()
        if not np.isnan(avg):
          ent_avg[entity_labels == label] = avg

    index_1 = torch.from_numpy(edge_array[:,0]).to(x_0.device)
    index_2 = torch.from_numpy(edge_array[:,1]).to(x_0.device)

    x = x_0.clone()
    u = u_0.clone()
    # Local averaging
    x_agg = scatter(x[index_2], index_1, dim_size=len(x)) # sum the neighbors
    r = scatter(u[index_2].type(torch.DoubleTensor).to(u.device), index_1, dim_size=len(x)) # count the known neighbors
    prim_nodes = (r>0) & (u==0)# indicates if an unknown node has neighbor with a known value
    x[prim_nodes] = x_agg[prim_nodes]/r[prim_nodes] # get the average
    u[prim_nodes] = 1
    if (u==0).any():
      x[u==0] = ent_avg[u==0]
    return x

def clamp(x, x_0, u_0):
    r"""
    Clamp the initialy known values

    x = torch.tensor(N,1)
    x_0 : torch.tensor of initial values with unkown zero-padded
    u_0 : torch.tensor of indicator of known values
    """
    x[u_0 == 1] = x_0[u_0 == 1]
    return x

def iter_MrAP(x_0, u_0, model, xi = 0.5, entity_labels = None):
  r"""
  Learn unkonwns over the iterations of MrAP

  x_0 : torch.tensor of initial values with unkown zero-padded
  u_0 : torch.tensor of indicator of known values
  model : MrAP() object indicating known/unknown and label propagation params
  xi: damping factor
  x_gt : numpy.ndarray grountruth value
  feasible_nodes: list/array of indices: eg; x_gt = x_0[feasible_nodes]
  entity_labels : numpy.array(object)

  returns prediction as torch.tensor()
  """
  x = x_0.clone()
  u = u_0.clone()
  eps = x.abs().max().item()/1000
  i = 0
  x_prev = torch.zeros_like(x)
  while ((x-x_prev).abs() > eps).any():
    x_prev = x.clone()
    x, u = model.forward(x, u, xi)
    x = clamp(x, x_0, u_0)
    i = i + 1
  if (u==0).any(): # check isolated components
    if entity_labels is None:
      x[u==0] = (x[u == 1]).mean().item() #average of the propagated values
    else:
      for label in np.unique(entity_labels[u.cpu().data.numpy()==0]):
        avg = x[u.cpu().data.numpy() & (entity_labels == label)].mean().item() # average of the propagated value in those entities
        if not np.isnan(avg):
          x[[u.cpu().data.numpy()==0] & (entity_labels == label)] = avg
        else:
          x[[u.cpu().data.numpy()==0] & (entity_labels == label)] = (x[u == 1]).mean().item()#if there is not any known entity, global average
  return x
