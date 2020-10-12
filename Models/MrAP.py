import numpy as np
import torch
from torch_scatter import scatter

class MrAP():
  def __init__(self, device, edge_list, eta = 1, omega = 1, tau = 0):
    r"""
    Construct multi-relational propagation model object for the given edge_list

    device = cpu or cuda:0
    N: # of nodes E : # of edges P: # of relation types
    edge_list = list of numpy.ndarray(E,2), each element corresponds to one type of relation
    eta = numpy.ndarray(P,1)
    omega = numpy.ndarray(P,1)
    tau = numpy.ndarray(P,1)

    self.index_1 : first column of edge pair array
    self.index_2 : second column of edge pair array
    self.t : content of matrix T
    self.s : content of matrix S
    self.h : content of matrix H
    """

    edge_array = np.concatenate(edge_list) # merge
    edge_array = np.concatenate((edge_array, np.flip(edge_array,1))) # [pointing out from index_1, pointing in to index_1]
    self.index_1 = torch.from_numpy(edge_array[:,0]).to(device)
    self.index_2 = torch.from_numpy(edge_array[:,1]).to(device)

    P = len(edge_list) # number of relation types
    if not isinstance(eta, np.ndarray): eta = eta*np.ones(P)
    if not isinstance(omega, np.ndarray): omega = omega*np.ones(P)

    div_omega = np.concatenate(list(np.repeat(1/omega[p], edge_list[p].shape[0]) for p in range(P))) # merge
    t_in = div_omega * np.concatenate(list(np.repeat(eta[p], edge_list[p].shape[0]) for p in range(P)))
    self.t = torch.from_numpy(np.concatenate((t_in,t_in))).to(device) # duplicate

    if not isinstance(tau, np.ndarray): tau = tau*np.ones(P)
    s_in = div_omega * np.concatenate(list(np.repeat(tau[p], edge_list[p].shape[0]) for p in range(P)))
    s_out = s_in * np.concatenate(list(np.repeat(eta[p], edge_list[p].shape[0]) for p in range(P))) # merge
    self.s = torch.from_numpy(np.concatenate((-s_out,s_in))).to(device) # duplicate

    h_out = div_omega * np.concatenate(list(np.repeat(eta[p]**2, edge_list[p].shape[0]) for p in range(P)))
    self.h = torch.from_numpy(np.concatenate((h_out,div_omega))).to(device) # duplicate

  def forward(self, x, u, xi=0):
    r"""
    One step forward pass

    x = torch.tensor(N,1)
    u = torch.tensor(N,1)
    xi : [0,1] damping factor (proportion to be updated by neighbors)
    """
    # Step-1 : Aggregate (Aggregation to index_1)
    x_agg = scatter(x[self.index_2]*self.t, self.index_1, dim_size=len(x))
    # Step-2 : Shift
    x_agg = x_agg + scatter(u[self.index_2]*self.s, self.index_1, dim_size=len(x))
    # Step-3 : Aggregate Neighbor weights
    r = scatter(u[self.index_2]*self.h, self.index_1, dim_size=len(x))
    # Step-4 : Normalize
    has_neighbor = r>0 # indicates if a node has neighbor with a known value
    x_agg[has_neighbor] = x_agg[has_neighbor]/r[has_neighbor]
    # Step-5 : Update
    # Update values of nodes propagated for the first time
    prim_nodes = (u==0) & has_neighbor
    x[prim_nodes] = x_agg[prim_nodes]
    # Damping for the nodes already with a known value
    if xi > 0 :
      known_nodes = (u==1) & has_neighbor
      x[known_nodes] = (1-xi)*x[known_nodes] + xi*x_agg[known_nodes]
    # Step-6 : Update indicator vector
    u[prim_nodes] = 1
    return x, u
