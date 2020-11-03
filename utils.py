import numpy as np
import torch

def performance(x_pred, x_gt, u_0, feasible_nodes = None):
    r"""
    Measure performance

    x_pred = torch.tensor(N,1), predicted tensor
    x_gt = numpy.ndarray(N,1) or torch.tensor(N,1) on device, ground-truth
    u_0 = torch.tensor(N,1), indicator of known/unknown

    output :
    RMSE (mean square error)
    MAE (mean square error)
    """
    if not feasible_nodes is None:
      x_pred = x_pred[feasible_nodes]
      u_0 = u_0[feasible_nodes]

    unknown_nodes = (u_0 == 0)
    if isinstance(x_gt, np.ndarray): x_gt = torch.from_numpy(x_gt).to(x_pred.device)
    x_gt = x_gt[unknown_nodes]
    x_pred = x_pred[unknown_nodes]

    RMSE = (((x_pred - x_gt)**2).mean()).sqrt()

    MAE = (x_pred - x_gt).abs()
    MAE = MAE.mean()

    return RMSE.item(), MAE.item()

def estimate_params(edge_list, x, rel_to_scale = None, u_0 = None):
  r"""
  edges = list of numpy.ndarray(E,2)
  x = numpy.ndarray(N,1), values that the correlation to be measured
  u_0 = numpy.ndarray(N,1), boolean indicator vector showing known values
  rel_to_scale = list of indices of relations need eta estimation

  return:
  tau = numpy.ndarray(P)
  omega = numpy.ndarray(P)
  eta = numpy.ndarray(P)
  """
  if u_0 is None:
    u_0 = np.ones_like(x, dtype=bool)
  if rel_to_scale is not None:
    from scipy import stats

  P = len(edge_list)
  tau = np.empty(P)
  omega = np.empty(P)
  eta = np.ones(P)
  diff_on_edges = []
  invalid_rel = []
  for p in range(P):
    edges = edge_list[p]
    known_pairs = u_0[edges[:,1]] & u_0[edges[:,0]]
    if sum(known_pairs) > 1:
      x_pairs = x[edges[known_pairs]]
      if rel_to_scale is not None:
        if p in rel_to_scale: eta[p], _, _, _, _ = stats.linregress(x_pairs[:,0], x_pairs[:,1])
      diff_on_edges.append(x_pairs[:,1] - eta[p]*x_pairs[:,0])
      tau[p] = diff_on_edges[p].mean()
      omega[p] = diff_on_edges[p].var()
    else:
      # print('Not enough known pairs to estimate tau and omega (0), default return!')
      diff_on_edges.append(0)
      tau[p] = 0
      omega[p] = 0
      invalid_rel.append(p)
  max_omega = np.max(omega)
  for p in invalid_rel:
    omega[p] = max_omega
  omega[omega == 0] = 1
  return tau, omega, eta, diff_on_edges

def drop_sym(edge_list):
  r"""
  reduce the duplicated undirected edges

  edge_list: list of numpy.ndrarray(E,2)

  return reduced edges as list of numpy.ndrarray(E,2)
  """
  asym_edge_list = []
  for edges in edge_list:
    edge_set = set(frozenset(item) for item in edges.tolist())
    asym_edges = np.array([np.array(list(item)) for item in edge_set])
    if len(asym_edges) == len(edges)/2:
      edges = asym_edges
    asym_edge_list.append(edges)
  return asym_edge_list

def reduce_to_singles(edge_list, entity_labels):
  r"""
    Reduce the edge list by removing cross-entity edges without changing the indexing of the nodes

    edge_list: list of numpy.array of edges
    entity_labels : numpy.array(object)

    output:
    edge_list_r : reduced edge_list
    rel_ind : indices for relations between the same type of entities
    entity_coupled : type of the entity coupled in each item in edge_list_r
    """
  edge_list_r = []
  entity_coupled = []
  rel_ind = []
  for i in range(len(edge_list)):
    egde2label = entity_labels[edge_list[i]]
    edges = edge_list[i][np.where(egde2label[:,0] == egde2label[:,1])[0], :]
    if edges.size > 0:
      edge_list_r.append(edges)
      rel_ind.append(i)
      entity_coupled.append(entity_labels[edges[0,0]])
  return edge_list_r, rel_ind, entity_coupled

def extract_edges_YAGO(triples, entities, inner_edge = True):
  attributes = entities.attribute.value_counts().keys().tolist()
  from itertools import combinations
  att_combs = list(combinations(attributes,2)) # attribute combinations
  dir_combs = [(item,item) for item in attributes] + att_combs +  [(item[1],item[0]) for item in att_combs]# directional combinations
  att_ent_dict = dict(zip(attributes,[entities[entities.attribute==item] for item in attributes]))

  edge_list = []
  relations = []
  # Single & Cross entity edges
  for comb in dir_combs:
    ent_1 = att_ent_dict[comb[0]]
    ent_2 = att_ent_dict[comb[1]]
    subset_triples = triples[triples.node_1.isin(ent_1['node'].tolist()) &
                             triples.node_2.isin(ent_2['node'].tolist())].copy()
    if len(subset_triples)>0:
      # print('Single or Cross:', comb, len(subset_triples))
      subset_triples.node_1 = subset_triples.node_1.map(lambda node_id: ent_1[ent_1.node == node_id].index.item())
      subset_triples.node_2 = subset_triples.node_2.map(lambda node_id: ent_2[ent_2.node == node_id].index.item())
      for rel in subset_triples.relation.unique():
        edge_list.append(subset_triples[subset_triples.relation == rel][['node_1','node_2']].to_numpy())
        relations.append(rel)
        # print('----------',rel,len(subset_triples[subset_triples.relation == rel]))

  # Inner entity edges
  if inner_edge:
    for comb in att_combs:
      overlap = entities[entities.attribute.isin(comb)]['node'].value_counts() > 1
      if any(overlap):
        # print('Inner:', comb, sum(overlap))
        pairs = []
        ent_1 = att_ent_dict[comb[0]]
        ent_2 = att_ent_dict[comb[1]]
        for node_id in overlap[overlap].keys().tolist():
          pairs.append([ent_1[ent_1.node == node_id].index.item(), ent_2[ent_2.node == node_id].index.item()])
        edge_list.append(np.array(pairs))
        relations.append(comb[0] + '_' + comb[1] + '_coupling')

  return edge_list, relations

def find_prim_sec_atts(entities):
  """ Find primary secondary attribute types
  input
  entities: pd.DataFrame with columns ['node', 'attribute', 'numeric']

  output:
  prim_att_list : list of primary attributes
  sec_prim_tuples : list of tuples paring each secondary to the corresponding primary attribute
  """

  def find_coexist_atts(entities):
    """ Find the multiple attributes existing in the same type of entity
    """
    att_coexist = []
    for item in entities.attribute.unique():
      node_interest = entities[entities.attribute == item]['node']
      atts = set(entities[entities.node.isin(node_interest)]['attribute'].tolist()) #co-existing attributes
      not_found = True
      for i in range(len(att_coexist)):
        if att_coexist[i].intersection(atts):
          att_coexist[i] = att_coexist[i].union(atts) # merge the attribute set
          not_found = False
          break
      if not_found:
        att_coexist.append(atts)
    return att_coexist

  att_coexist = find_coexist_atts(entities)
  prim_att_list = [] # Find attributes in majority at each entity type (primal attributes)
  sec_prim_tuples = [] # Find the minorities (seconder) and the corresponding primal
  # print('Multiple attributes existing on same entity:')
  for item in att_coexist:
    # print(item)
    item = list(item)
    if len(item)>1: # multiple attributes, there are secondaries
      att_counts = [sum(entities.attribute== each_att) for each_att in item]
      prim_att = item[att_counts.index(max(att_counts))] # attribute in majority among the multiple
      item.remove(prim_att) # rest is secondary
      sec_prim_tuples = sec_prim_tuples + [(sec_att, prim_att) for sec_att in item]
    else:
      prim_att = item[0]
    prim_att_list.append(prim_att)
  return prim_att_list, sec_prim_tuples

def extract_edges_FB(triples, entities, inner_edge = True):
  """ Compose list of edges connecting attributes by inner and outer edges
  triples : pd.DataFrame with columns ['node_1', 'relation', 'node_2']
  entities: pd.DataFrame with columns ['node', 'attribute', 'numeric']
  inner_edge: Also include the primary secondary pairing edges if True

  output:
  edge_list: list of numpy.array of edges
  relations: list of relation types
  """

  prim_att_list, sec_prim_tuples = find_prim_sec_atts(entities)
  # Extract the edges
  edge_list = []
  relations = []

  def triples_entities_to_edgelist(triples, entities, edge_list, relations):
    """ Extract the edges for given entities and enlarge the edge_list and relations
    """
    triples.node_1 = triples.node_1.map(lambda node_id: entities[entities.node == node_id].index.item())
    triples.node_2 = triples.node_2.map(lambda node_id: entities[entities.node == node_id].index.item())
    for rel in triples.relation.unique():
      edge_list.append(triples[triples.relation == rel][['node_1','node_2']].to_numpy())
      relations.append(rel)
    return edge_list, relations

  # Edge list between primaries -- OUTER
  subset_entities = entities[entities.attribute.isin(prim_att_list)] #all nodes are unique in this subset
  nodes = subset_entities.node.tolist()
  subset_triples = triples[triples.node_1.isin(nodes) & triples.node_2.isin(nodes)].copy()
  egde_list, relations = triples_entities_to_edgelist(subset_triples, subset_entities, edge_list, relations)

  # Edge list connecting secondaries
  for item in sec_prim_tuples: # item=(sec,prim)
    # replace the corresponding primary with secondary
    prim_rest = prim_att_list.copy()
    prim_rest.remove(item[1])
    atts_interest = prim_rest + [item[0]]
    subset_entities = entities[entities.attribute.isin(atts_interest)] #all nodes are unique in this subset
    nodes = subset_entities.node.tolist()
    nodes_sec = entities[entities.attribute == item[0]]['node'].tolist()
    subset_triples = triples[(triples.node_1.isin(nodes) & triples.node_2.isin(nodes_sec)) |
                              (triples.node_1.isin(nodes_sec) & triples.node_2.isin(nodes))].copy()
    # edges to other type of entities -- OUTER
    egde_list, relations = triples_entities_to_edgelist(subset_triples, subset_entities, edge_list, relations)

    # coupling edges to the corresponding primary -- INNER
    if inner_edge:
      pairs = []
      for node in nodes_sec: # node carrying secondary
        if node in entities[entities.attribute== item[1]]['node'].tolist(): # also have the primary
          pairs.append([entities[(entities.node==node) & (entities.attribute == item[0])].index.item(),
                        entities[(entities.node==node) & (entities.attribute == item[1])].index.item()])
      egde_list.append(np.array(pairs))
      relations.append(item[0] + '_coupling')

  return edge_list, relations
