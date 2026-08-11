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
            gap_lengths = np.array([]) # No internal gaps if there is 1 or 0 blocks
    
        # Feauture dictionary for the current sequence
        metrics = {
            'num_blocks': len(block_lengths), #number of blocks of amino acids from the group in the sequence
            'mean_block_len': np.mean(block_lengths) if len(block_lengths) > 0 else 0.0, #mean length of blocks of amino acids from the group in the sequence
            'var_block_len': np.var(block_lengths) if len(block_lengths) > 0 else 0.0, #variance of the length of blocks of amino acids from the group in the sequence
            'max_block_len': np.max(block_lengths) if len(block_lengths) > 0 else 0.0, #maximum length of all the blocks of amino acids from the group in the sequence
            'num_gaps': len(gap_lengths), #number of gaps between blocks of amino acids from the group in the sequence
            'mean_gap_len': np.mean(gap_lengths) if len(gap_lengths) > 0 else 0.0, #mean length of gaps between blocks of amino acids from the group in the sequence
            'var_gap_len': np.var(gap_lengths) if len(gap_lengths) > 0 else 0.0, #variance of the length of gaps between blocks of amino acids from the group in the sequence
            'max_gap_len': np.max(gap_lengths) if len(gap_lengths) > 0 else 0.0 #maximum length of all the gaps between blocks of amino acids from the group in the sequence
        }
        metrics_list.append(metrics)
        
    return pd.DataFrame(metrics_list)


'''def interaction_map_coefficient(sequence_list, potential='MIYS960102'): #MIYS960102 #TANS760101
    
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        potential : str
            The potential to be used for the interaction map from the aaindex database.
    Output:
        Returns a Dataframe in which each row is a sequence and the columns
        are the matrix and the features of the LxL interaction map matrix. (Trace, Determinant, Sum, Min_Eigenvalues, Max_Eigenvalues)
    
    from aaindex import aaindex3

    # Init results
    metrics_list = []

    record = aaindex3[potential]
    matrix_data = record['matrix']

    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    MJ_matrix = np.zeros((20, 20))
    for i, aa1 in enumerate(AMINO_ACIDS):
        for j, aa2 in enumerate(AMINO_ACIDS):
            MJ_matrix[i, j] = matrix_data[aa1][aa2]

    # Operate for each sequence
    for s in sequence_list:
        

        def get_sequence_interaction_matrix(sequence, base_matrix):
            """
            Given a sequence of length L, returns an L x L matrix
            containing the corresponding interaction values.
            """

            indices = [aa_to_idx[aa.upper()] for aa in sequence]
            idx_array = np.array(indices)
            
            L_x_L_matrix = base_matrix[idx_array][:, idx_array]
            return L_x_L_matrix

        M = get_sequence_interaction_matrix(s, MJ_matrix)

        metrics = {
            #'Matrix': M,
            'Trace': np.trace(M),
            'Determinant': np.linalg.det(M),
            'Sum': np.sum(M),
            #'Eigenvalues': np.linalg.eigvals(M)
            'Min_Eigenvalue': np.min(np.linalg.eigvals(M)),
            'Max_Eigenvalue': np.max(np.linalg.eigvals(M))
        }

        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)'''


def KL_divergence_coefficient(sequence_list,amino_group):
    """
    Computes the KL divergence coefficient for a list of sequences based on the specified amino acid group.

    Parameters:
    sequence_list (list): A list of sequences (strings) to analyze.
    amino_group (list): A list of amino acids representing the group of interest.

    Returns:
    res (list): A list of KL divergence coefficients for each sequence and amino acid group.
    """
    # Initialize counts
    total_count = 0
    group_count = 0

    # Count occurrences of amino acids in the specified group in all sequences
    for s in sequence_list:
        total_count += len(s)
        group_count += sum(1 for aa in s if aa in amino_group)

    # Calculate probabilities
    p_group_total = group_count / total_count if total_count > 0 else 0
    p_not_group_total = 1 - p_group_total
    res = []

    for s in sequence_list:
        seq_count = len(s)
        seq_group_count = sum(1 for aa in s if aa in amino_group)

        # Calculate probabilities for the current sequence
        p_group = seq_group_count / seq_count if seq_count > 0 else 0
        p_not_group = 1 - p_group
        # Calculate KL divergence coefficient for the current sequence wrt the total distribution
        kl_divergence = 0
        if p_group > 0:
            kl_divergence += p_group * np.log2(p_group / (p_group_total))
        if p_not_group > 0:
            kl_divergence += p_not_group * np.log2(p_not_group / (p_not_group_total))

        res.append(kl_divergence)


    return res


def signal_processing_metrics(sequence_list,amino_group1,amino_group2):
    '''
    Params:
        sequence_list : list
            List containing the sequences to be analyzed.
        amino_group1 : list
            List containing the amino acids of the first group.
        amino_group2 : list
            List containing the amino acids of the second group.
    Output:
        Returns a Dataframe in which each row is a sequence and the columns
        are the features extracted from the signal processing analysis (lag_minus_one, lag_plus_one, max_xcorr, max_power_one, max_power_two).
    '''
    # Init results
    res = []
    
    for s in sequence_list:
        # Convert sequence to binary signal
        seq_vec = np.array(list(s))
        v = 1*np.isin(seq_vec, amino_group1)
        w = 1*np.isin(seq_vec, amino_group2)

        xcorr = np.correlate(v, w, mode='full')
        xcenter = len(s) - 1

        lag_minus_one = xcorr[xcenter - 1] if (xcenter - 1) >= 0 else 0
        lag_plus_one = xcorr[xcenter + 1] if (xcenter + 1) < len(xcorr) else 0

        max_xcorr = np.max(xcorr)
        argmax_xcorr = np.argmax(xcorr)

        max_power_one = np.max((np.abs(np.fft.fft(v))**2)[1:])  
        max_power_two = np.max((np.abs(np.fft.fft(w))**2)[1:]) 

        metrics = {
            'lag_minus_one': lag_minus_one, # value of the cross-correlation at lag -1
            'lag_plus_one': lag_plus_one, # value of the cross-correlation at lag +1
            'max_xcorr': max_xcorr, # maximum value of the cross-correlation
            'argmax_xcorr': argmax_xcorr, # delay at which the maximum cross-correlation occurs
            'max_power_one': max_power_one, # maximum power of the first signal frequency (amino_group1)
            'max_power_two': max_power_two # maximum power of the second signal frequency (amino_group2)
        }
        res.append(metrics)

    return pd.DataFrame(res)



# ----- OTHER ------
# Typical grouping of the standard amino acids
# Define amino acid groups
amino_groups_standard = {'Hydrophobic': ['C', 'M', 'V', 'L', 'I'],
 'Aromatic': ['W', 'Y', 'F'],
 'Hydrophilic': ['H', 'T', 'N', 'Q'],
 'Charged+': ['K', 'R'],
 'Charged-': ['D', 'E'],
 'Disorder promoting': ['A', 'S', 'G', 'P']}