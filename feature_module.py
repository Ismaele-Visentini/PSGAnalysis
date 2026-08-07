# Import modules
import pandas as pd
import numpy as np



# ------ I/O FUNCTIONS -------





# ------ FEATURE FUNCTIONS ------
# Function to calculate difference in amino acid composition
def group_composition(sequence_list, amino_group, normalize=False):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure total count.
        normalize : bool (optional)
            Whether to normalize the results by the length
            of each sequence or not. Default is False.
    Output:
        Returns a list of values, indicating the amount
        of amino acids (or the fraction if normalize=True)
        from the given group in each input sequence.
    '''
    # Init results
    res = []

    # Operate for each sequence
    for s in sequence_list:
        s_arr = pd.Series(list(s))
        
        count = np.sum(s_arr.isin(amino_group))
        if normalize:
            count = count / len(s)

        res.append(count)

    return res


# Function to calculate the adjacency pairs coefficient
def adjacent_pairs_coefficient(sequence_list, amino_group, normalize=True):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure adjacency.
        normalize : bool (optional)
            Whether to normalize the results by the length
            of each sequence or not. Default is True.
    Output:
        Returns a list of values, indicating the adajcency
        pairs coefficient (or the count if normalize=False)
        from the given group in each input sequence.
    '''
    # Init results
    res = []

    # Operate for each sequence
    for s in sequence_list:
        # Build binary vector
        seq_vec = np.array(list(s))
        v = 1*np.isin(seq_vec, amino_group)

        # Build n+1 vector
        v_next = np.zeros(v.shape)
        v_next[1:] += v[:-1]

        # Compute adjacent pairs
        A = np.sum(v * v_next)

        # Normalize and return
        A_norm = A / (np.sum(v) - 1)

        if not normalize:
            res.append(A)
        else:
            res.append(A_norm)

    return res


# Function to calculate the adjacency triplets coefficient
def adjacent_triplets_coefficient(sequence_list, amino_group, normalize=True):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure adjacency.
        normalize : bool (optional)
            Whether to normalize the results by the length
            of each sequence or not. Default is True.
    Output:
        Returns a list of values, indicating the adajcency
        triplets coefficient (or the count if normalize=False)
        from the given group in each input sequence.
    '''
    # Init results
    res = []

    # Operate for each sequence
    for s in sequence_list:
        # Build binary vector
        seq_vec = np.array(list(s))
        v = 1*np.isin(seq_vec, amino_group)

        # Build n+1 and n+2 vectors
        v_next = np.zeros(v.shape)
        v_next[1:] += v[:-1]

        v_trp = np.zeros(v.shape)
        v_trp[2:] += v[:-2]

        # Compute adjacent triplets
        A = np.sum(v * v_next * v_trp)

        # Normalize and return
        A_norm = A / (np.sum(v) - 2)

        if not normalize:
            res.append(A)
        else:
            res.append(A_norm)

    return res


# Function to calculate the adjacency quadruplets coefficient
def adjacent_quadruplets_coefficient(sequence_list, amino_group, normalize=True):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure adjacency.
        normalize : bool (optional)
            Whether to normalize the results by the length
            of each sequence or not. Default is True.
    Output:
        Returns a list of values, indicating the adajcency
        quadruplets coefficient (or the count if normalize=False)
        from the given group in each input sequence.
    '''
    # Init results
    res = []

    # Operate for each sequence
    for s in sequence_list:
        # Build binary vector
        seq_vec = np.array(list(s))
        v = 1*np.isin(seq_vec, amino_group)

        # Build n+1 and n+2 vectors
        v_next = np.zeros(v.shape)
        v_next[1:] += v[:-1]

        v_trp = np.zeros(v.shape)
        v_trp[2:] += v[:-2]

        v_quad = np.zeros(v.shape)
        v_quad[3:] += v[:-3]

        # Compute adjacent triplets
        A = np.sum(v * v_next * v_trp * v_quad)

        # Normalize and return
        A_norm = A / max([(np.sum(v) - 3),1])

        if not normalize:
            res.append(A)
        else:
            res.append(A_norm)

    return res

def gap_coefficient(sequence_list, amino_group):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure adjacency.
    Output:
        Returns a list of values, indicating the gap coefficient
        from the given group in each input sequence.
    '''
    # Init results
    res = []
    mean_list = []
    var_list = []
    d_list = []
    # Operate for each sequence
    for s in sequence_list:
        # Build binary vector
        seq_vec = np.array(list(s))
        v = 1*np.isin(seq_vec, amino_group)
        v = np.diff(v) #v[i+1]-v[i]
        s = 1*np.isin(v,-1) #1 if start, 0 if not
        f = 1*np.isin(v, 1) #1 if finish, 0 if not
        d = s-f
        d_list.append(d)
        mean_list.append(np.mean(d))
        var_list.append(np.var(d))
    return d_list, mean_list, var_list

def block_distribution_metrics(sequence_list, amino_group):
    '''
     Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group : list
            List containing the amino acids for which to
            measure adjacency.
    Output:
        Returns a DataFrame where each row is a sequence and the columns
        are the features of block and gap distribution. (num blocks, mean block length, var block length, max block length, mean gap length, var gap length, max gap length)
    '''
    metrics_list = []
    amino_array = np.array(amino_group)
    
    for s in sequence_list:
        seq_vec = np.array(list(s))
        
        #Binary vector: 1 if in the group, 0 otherwise
        v = np.isin(seq_vec, amino_array).astype(int)
        
        # Adding padding of zeros at the ends to correctly capture blocks that start at the first character or end at the last
        padded_v = np.pad(v, (1, 1), mode='constant')
        diffs = np.diff(padded_v)
        
        #Find the indices of starts and ends of blocks
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        #1. Calculate the lengths of the blocks (distance between start and end indices)
        block_lengths = ends - starts
        
        #2. Calculate the lengths of the internal gaps (distance between the end of one block and the start of the next)
        if len(starts) > 1:
            gap_lengths = starts[1:] - ends[:-1]
        else:
            gap_lengths = np.array([]) # Nessun gap interno se c'è 1 o 0 blocchi
    
        # Feauture dictionary for the current sequence
        metrics = {
            'num_blocks': len(block_lengths),
            'mean_block_len': np.mean(block_lengths) if len(block_lengths) > 0 else 0.0,
            'var_block_len': np.var(block_lengths) if len(block_lengths) > 0 else 0.0,
            'max_block_len': np.max(block_lengths) if len(block_lengths) > 0 else 0.0,
            'mean_gap_len': np.mean(gap_lengths) if len(gap_lengths) > 0 else 0.0,
            'var_gap_len': np.var(gap_lengths) if len(gap_lengths) > 0 else 0.0,
            'max_gap_len': np.max(gap_lengths) if len(gap_lengths) > 0 else 0.0
        }
        metrics_list.append(metrics)
        
    return pd.DataFrame(metrics_list)

# ----- OTHER ------
# Typical grouping of the standard amino acids
# Define amino acid groups
amino_groups_standard = {'Hydrophobic': ['C', 'M', 'V', 'L', 'I'],
 'Aromatic': ['W', 'Y', 'F'],
 'Hydrophilic': ['H', 'T', 'N', 'Q'],
 'Charged+': ['K', 'R'],
 'Charged-': ['D', 'E'],
 'Disorder promoting': ['A', 'S', 'G', 'P']}