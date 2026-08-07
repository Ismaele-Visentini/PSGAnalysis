# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------- DATA PROCESSING FUNCTIONS --------
def gather_feature_pair_samples(feature_1, feature_2, metric, cutoff):
    '''
    Params:
        feature_1, feature_2 : list or numpy array
            Features 1 and 2 to use for the x and y axes, respectively,
            of the 2D grid.
        metric : list or numpy array
            Metric to use to color the points in the 2D grid.
        cutoff : int
            Cutoff value to impose on the (feature_1, feature_2, metric)
            tuples in order to remove low-confidence samples.
    Output:
        Returns the unique tuples of points (feature_1, feature_2, s),
        where s is the average of the given metric over all samples that
        correspond to (feature_1, feature_2).
    '''
    # Gather data
    x_data = np.array(feature_1).flatten()
    y_data = np.array(feature_2).flatten()
    s_data = np.array(metric).flatten()
    x_vals = []
    y_vals = []
    s_vals = []
    for k1 in np.unique(x_data):
        k1_mask = x_data == k1
        for k2 in np.unique(y_data):
            k2_mask = y_data == k2
            if np.sum(k1_mask * k2_mask) > cutoff:
                # Save xy pair
                x_vals.append(k1)
                y_vals.append(k2)
    
                # Get mean score
                s_vals.append(np.mean(s_data[k1_mask & k2_mask]))
    
    return x_vals, y_vals, s_vals


# ------- ANALYSIS FUNCTIONS --------
# Function to calculate the partial correlation between variables
def partial_corr_manual(df):
    '''
    Params:
        df : pandas DataFrame
            DataFrame containing the variables to be analyzed as 
            columns.
    Output:
        Returns the partial correlation matrix between the given
        variables.
    '''
    # Calculate correlation matrix
    corr_matrix = df.corr()
    # Inverse of the correlation matrix
    corr_inv = np.linalg.inv(corr_matrix)
    
    # Initialize partial correlation matrix
    p_corr = np.zeros_like(corr_matrix)
    
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if i == j:
                p_corr[i, i] = 1.0
            else:
                # Formula: -Omega_ij / sqrt(Omega_ii * Omega_jj)
                p_corr[i, j] = -corr_inv[i, j] / np.sqrt(corr_inv[i, i] * corr_inv[j, j])
                
    return pd.DataFrame(p_corr, index=df.columns, columns=df.columns)


# Function to calculate the weighted standard deviation
def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    # For sample standard deviation (unbiased), apply Bessel's correction
    # variance = variance * sum(weights) / (sum(weights) - 1)
    return np.sqrt(variance)



# Function to get the partial correlation for feature pairs
def feature_pair_partial_correlation(feature_1, feature_2, metric, cutoff):
    '''
    Params:
        feature_1, feature_2 : list or numpy array
            Features 1 and 2 to use as independent variables.
        metric : list or numpy array
            Metric to use as dependent variable.
        cutoff : int
            Cutoff value to impose on the (feature_1, feature_2, metric)
            tuples in order to remove low-confidence samples.
    Output:
        Returns a pandas DataFrame with the partial correlations among
        the given variables.
    '''
    # Gather samples
    x_vals, y_vals, s_vals = gather_feature_pair_samples(
        feature_1, feature_2, metric, cutoff)
    df = pd.DataFrame({'Feature 1': x_vals, 'Feature 2': y_vals, 'Metric': s_vals})
    
    # Calculate partial correlation and return
    partial_corr = partial_corr_manual(df)
    return partial_corr



# ------- VISUALIZATION FUNCTIONS -------
# Function to plot a single variable correlation between a feature and a metric
def plot_single_feature_metric(feature, metric, bins=12, fontsize=13, lw=2, cs=3,
                               figsize=(5,3), xlabel='Feature', ylabel='Metric',
                               path=None, sigmas=1):
    '''
    Params:
        feature : list or numpy array
            Feature values to use as independent variable.
        metric : list or numpy array
            Metric to use as dependent variable, must have
            the same length as feature.
        bins : int (optional)
            Number of bins to use for the numpy histogram2d 
            function. Default value is 12.
        fontsize : int (optional)
            Fontsize for the plot elements. Default is 13.
        lw : int of float (optional)
            Line width for the plot curves. Default is 2.
        cs : int or float (optional)
            Capsize for the plot errorbars. Default is 3.
        figsize : 2-tuple (optional)
            Size of the plot figure. Default is (5,3).
        xlabel : str (optional)
            Label for the x-axis of the plot. Default is
            "Feature".
        ylabel : str (optional)
            Label for the y-axis of the plot. Default is
            "Metric".
        path : str (optional)
            Path to save the plot as a file. Not used by 
            default.
        sigmas : int (optional)
            How many standard deviations to show in the
            errorbars. Default is 1.
    Output:
        Returns a plot of the metric as a function of the
        given feature, produced by binning the variables 
        and aggegating the metric over the feature bins.
    '''
    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, dpi=200)

    # Plot results
    xvals = np.array(feature).flatten()
    yvals = np.array(metric).flatten()
    hist = np.histogram2d(xvals, yvals, bins=bins, density=True)

    # Plot y mean 
    slices = 0.5 * (np.array(hist[2][1:]) + np.array(hist[2][:-1]))
    xslices = 0.5 * (np.array(hist[1][1:]) + np.array(hist[1][:-1]))
    ymean = np.average(slices, weights=np.sum(hist[0],axis=0))
    ymean = [ymean for l in range(bins)]
    ax.plot(xslices, ymean, c='red', ls=':', lw=lw)

    # Plot mean per vertical slice
    yavg = []
    yerr = []
    for k in range(hist[0].shape[0]):
        weights = hist[0][k,:]
        if np.sum(weights) == 0:
            yavg.append(np.mean(yvals))
            yerr.append(sigmas * np.std(yvals) / np.sqrt(len(yvals)))
        else:
            yavg.append(np.average(slices, weights=weights))
            yerr.append(sigmas * weighted_std(slices, weights) / np.sqrt(len(slices)))
    ax.errorbar(xslices, yavg, yerr=yerr, c='red', lw=lw, capsize=cs)

    # Set labels
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)  
    ax.tick_params(labelsize=fontsize-2)
    #ax.set_xlim(left=np.min(xslices), right=np.max(xslices))

    plt.tight_layout()
    if path != None:
        plt.savefig(path, dpi=200)

    plt.show()
    plt.close();



# Function to plot a 2D grid of a feature pair against a metric
def plot_2D_feature_grid(feature_1, feature_2, metric, cutoff, cmap='viridis_r',
                         figsize=(5,3), xlabel='Feature 1', ylabel='Feature 2',
                         colbar_label='Metric', fontsize=13, path=None):
    '''
    Params:
        feature_1, feature_2 : list or numpy array
            Features 1 and 2 to use for the x and y axes, respectively,
            of the 2D grid.
        metric : list or numpy array
            Metric to use to color the points in the 2D grid.
        cutoff : int
            Cutoff value to impose on the (feature_1, feature_2, metric)
            tuples in order to remove low-confidence samples.
        cmap : str (optional)
            Colormap to use for the 2D grid. Default is viridis_r.
        figsize : 2-tuple (optional)
            Size of the plot figure. Default is (5,3).
        xlabel : str (optional)
            Label for the x-axis of the plot. Default is
            "Feature 1".
        ylabel : str (optional)
            Label for the y-axis of the plot. Default is
            "Feature 2".
        colbar_label : str (optional)
            Label for the adjacent colorbar. Default is
            "Metric".
        fontsize : int (optional)
            Fontsize for the plot elements. Default is 13.
        path : str (optional)
            Path to save the plot as a file. Not used by 
            default.
    Output:
        Returns a plot of a 2D grid composed of the features 1 and 2,
        with the unique tuples colored according to the given metric and
        filtered according to the given cutoff.
    '''
    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, dpi=200)
    
    x_vals, y_vals, s_vals = gather_feature_pair_samples(
        feature_1, feature_2, metric, cutoff)
            
    # Plot scatter and colorbar
    sctr = ax.scatter(x_vals, y_vals, c=s_vals, cmap=cmap)
    cbar = plt.colorbar(sctr)

    # Set labels
    cbar.ax.set_ylabel(colbar_label)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    
    plt.tight_layout()
    if path != None:
        plt.savefig(path, dpi=200)

    plt.show()
    plt.close();



# Function to perform and plot an exploration of different cutoff values
# for a partial correlation analysis
def plot_partial_correlation_vs_cutoff(feature_1, feature_2, metric, cutoff_values, 
                                       fontsize=13, lw=2, figsize=(5,3), path=None,
                                       feature_1_name='Feature 1', 
                                       feature_2_name='Feature 2'):
    '''
    Params:
         feature_1, feature_2 : list or numpy array
            Features 1 and 2 to use as independent variables.
        metric : list or numpy array
            Metric to use as dependent variable.
        cutoff : list
            Cutoff values to impose on the (feature_1, feature_2, metric)
            tuples in order to remove low-confidence samples.
        fontsize : int (optional)
            Fontsize for the plot elements. Default is 13.
        lw : int of float (optional)
            Line width for the plot curves. Default is 2.
        figsize : 2-tuple (optional)
            Size of the plot figure. Default is (5,3).
        path : str (optional)
            Path to save the plot as a file. Not used by 
            default.
        feature_1_name : str (optional)
            Name for feature 1 to use for the plot legend. Default is
            "Feature 1".
        feature_2_name : str (optional)
            Name for feature 2 to use for the plot legend. Default is
            "Feature 2".
    Output:
        Returns a plot with the partial correlation of the given features
        and the metric, with respect to the cutoff values used for the
        analysis.
    '''
    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1, dpi=200)

    # Get results
    corr_vals = {feature_1_name: [], feature_2_name: []}
    for cutoff in cutoff_values:
        x_vals, y_vals, s_vals = gather_feature_pair_samples(
            feature_1, feature_2, metric, cutoff)
        
        df = pd.DataFrame({feature_1_name: x_vals, feature_2_name: y_vals, 'Metric': s_vals})
        
        partial_corr = partial_corr_manual(df)
        plot_data = partial_corr['Metric'].drop('Metric')

        # Save data
        corr_vals[feature_1_name].append(plot_data[feature_1_name])
        corr_vals[feature_2_name].append(plot_data[feature_2_name])
        
    # Plot results
    ax.plot(cutoff_values, corr_vals[feature_1_name], ls='-', lw=lw, label=feature_1_name)
    ax.plot(cutoff_values, corr_vals[feature_2_name], ls='--', lw=lw, label=feature_2_name)
    ax.grid(True)
    ax.set_ylim(bottom=-1, top=1)

    # Set labels
    ax.set_ylabel('Partial correlation', fontsize=fontsize)
    ax.set_xlabel('Cutoff', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-2)
    plt.legend(fontsize=fontsize-2)

    plt.tight_layout()
    plt.show();