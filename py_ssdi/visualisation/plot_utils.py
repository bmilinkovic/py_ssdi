import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set default font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})


def plot_model_matrices(connectivity, A_var, C_ss, A_ss, V, K_ss):
    """
    Plot the VAR connectivity mask, VAR coefficient matrices, and SS parameters.

    Parameters
    ----------
    connectivity : ndarray, shape (n, n)
        Binary mask indicating allowed VAR connections per lag.
    A_var : list of ndarray, each shape (n, n, r_var)
        List of VAR coefficient matrices for each lag.
    C_ss : ndarray, shape (n, n*p)
        Observation matrix of the companion-form SS model.
    A_ss : ndarray, shape (n*p, n*p)
        State-transition (companion) matrix of the SS model.
    V : ndarray, shape (n, n)
        Residual covariance (should be identity after transform_var).
    K_ss : ndarray, shape (n*p, n)
        Kalman gain matrix of the SS model.
    """
    n, _, r = connectivity.shape

    # Plot connectivity mask
    plt.figure()
    sns.heatmap(connectivity[:, :, 0], cmap='bone_r', center=0,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Mask value'})
    plt.gca().invert_yaxis()  # Invert y-axis after creating heatmap
    plt.title('Prespecified connectivity mask')
    plt.xlabel('From variable')
    plt.ylabel('To variable')

    # Plot VAR coefficient matrices per lag
    for p in range(A_var.shape[2]):
        A_p = A_var[:, :, p]
        plt.figure()
        sns.heatmap(A_p, cmap='bone_r', center=0,
                    xticklabels=True, yticklabels=True,
                    cbar_kws={'label': 'Coefficient value'})
        plt.title(f'VAR coefficient A (lag {p+1})')
        plt.xlabel('From variable')
        plt.ylabel('To variable')

    # Plot SS matrices: C_ss, A_ss, V, K_ss
    titles = ['C_ss (observation)', 'A_ss (state-transition)', 
              'V (residual covariance)', 'K_ss (Kalman gain)']
    mats   = [C_ss, A_ss, V, K_ss]
    for title, M in zip(titles, mats):
        plt.figure()
        sns.heatmap(M, cmap='bone_r', center=0,
                    xticklabels=True, yticklabels=True,
                    cbar_kws={'label': 'Value'})
        plt.title(title)
        plt.xlabel('Column index')
        plt.ylabel('Row index')


def plot_optimisation_runs(histories):
    """
    Plot dynamical dependence over iterations for each optimisation run,
    highlighting the run with the minimum final DD.

    Parameters
    ----------
    histories : list of ndarray
        Each element is an array of shape (iters, 3), where column 0 = DD value.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Compute final DD for each run
    final_dd = [hist[-1, 0] for hist in histories]
    best_idx = int(np.argmin(final_dd))

    for idx, hist in enumerate(histories):
        dd_values = hist[:, 0]
        if idx == best_idx:
            ax.plot(dd_values, linewidth=2.0, label='Best run')
        else:
            ax.plot(dd_values, linewidth=0.5, alpha=0.5)

    ax.set_title('Preoptimisation: Dynamical dependence over iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('DD')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Ensure tight layout
    fig.tight_layout()


def plot_distance_matrix(goptp, show_title=True):
    """
    Plot the pairwise suboptimality distance matrix from preoptimisation.
    The colormap range is fixed from 0 to 1 to ensure consistency across plots.

    Parameters
    ----------
    goptp : ndarray, shape (N, N)
        Pairwise distance among the N preoptimisation runs.
        Values should be normalized to [0,1] range.
    show_title : bool, optional
        Whether to show the title (default: True)
    """
    fig, ax = plt.subplots(figsize=(8, 8))  # Square figure
    
    # Get matrix dimensions
    N = goptp.shape[0]
    
    # Select tick positions: first, middle, and last
    tick_positions = [0, N//2, N-1]
    tick_labels = [str(i+1) for i in tick_positions]
    
    # Create heatmap with square aspect ratio and fixed colormap range
    sns.heatmap(goptp, cmap='bone_r',
                vmin=0, vmax=1,  # Fix colormap range from 0 to 1
                xticklabels=tick_labels if tick_positions else False,
                yticklabels=tick_labels if tick_positions else False,
                cbar_kws={'label': 'Distance'},
                ax=ax,
                square=True)  # Force square cells
    
    ax.invert_yaxis()  # Invert y-axis after creating heatmap
    
    # Only show title if requested
    if show_title:
        ax.set_title('Preoptimisation subspace distance matrix (goptp)')
    
    ax.set_xlabel('Run index')
    ax.set_ylabel('Run index')
    
    # Remove minor ticks
    ax.tick_params(axis='both', which='minor', length=0)
    
    # Ensure tight layout
    fig.tight_layout()


def plot_composite_histories(results_dict):
    """
    Create a composite figure of all optimization histories.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing results for each m value.
        Expected structure: {'m=X': {'histp': [...], ...}, ...}
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    m_keys = sorted(results_dict.keys(), key=lambda x: int(x.split('=')[1]))
    n_m = len(m_keys)
    
    # Create figure with height proportional to number of subplots
    # Make plots taller relative to width (reduced width, increased height per plot)
    fig = plt.figure(figsize=(8, 3*n_m))
    
    # Create subplots with more vertical space between them
    gs = GridSpec(n_m, 1, figure=fig, hspace=0.4)
    
    for idx, m_key in enumerate(m_keys):
        ax = fig.add_subplot(gs[idx, 0])
        histories = results_dict[m_key].get('histp', [])
        
        if histories:
            # Plot each run's DD values on log-log scale
            for hist in histories:
                x = np.arange(1, len(hist)+1)
                y = hist[:, 0]
                ax.plot(x, y, linewidth=0.7, alpha=0.7)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f"Preopt DD histories ({m_key})")
            # Only show x-label on bottom subplot
            if idx == n_m - 1:
                ax.set_xlabel('Iteration (log)')
            ax.set_ylabel('DD')
            ax.grid(True, alpha=0.3)
    
    return fig


def plot_composite_goptp(results_dict):
    """
    Create a composite figure of all goptp matrices with a shared colorbar.
    Arranges plots in 2 columns, filling horizontally first (left to right, then down).
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing results for each m value.
        Expected structure: {'m=X': {'goptp': array, ...}, ...}
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    m_keys = sorted(results_dict.keys(), key=lambda x: int(x.split('=')[1]))
    n_m = len(m_keys)
    
    # Always use 2 columns
    n_cols = 2
    # Calculate number of rows needed
    n_rows = (n_m + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with GridSpec to accommodate colorbar at bottom
    fig = plt.figure(figsize=(6*n_cols, 6*n_rows + 0.2))  # Extra height for colorbar
    
    # Create GridSpec with proper spacing
    # Main grid for plots and a small horizontal space at bottom for colorbar
    gs = GridSpec(n_rows + 1, n_cols,  # Add one row for colorbar
                 height_ratios=[*[1]*n_rows, 0.1],  # Last row is smaller for colorbar
                 hspace=0.3, wspace=0.3)  # Reduced vertical spacing
    
    # Initialize norm for shared colorbar
    norm = plt.Normalize(vmin=0, vmax=1)
    
    # Fill plots horizontally first, then move to next row
    for idx, m_key in enumerate(m_keys):
        # Calculate position in grid (horizontal filling)
        row = idx // n_cols
        col = idx % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        goptp = results_dict[m_key].get('goptp', None)
        if goptp is not None:
            N = goptp.shape[0]
            
            # Create proper tick positions and labels
            tick_positions = [0, N//2, N-1]
            tick_labels = [1, N//2 + 1, N]  # Actual run numbers (1-based)
            
            # Create heatmap without individual colorbars
            sns.heatmap(goptp, cmap='bone_r',
                       vmin=0, vmax=1,
                       xticklabels=tick_labels,
                       yticklabels=tick_labels,
                       cbar=False,  # No individual colorbars
                       ax=ax,
                       square=True)
            
            ax.invert_yaxis()
            ax.set_title(f"m={m_key.split('=')[1]}", fontsize=20, pad=5)  # Even larger title
            
            # Ensure ticks are shown and properly positioned with larger font
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=16, rotation=0)  # Horizontal labels
            ax.set_yticklabels(tick_labels, fontsize=16)
            
            if col == 0:  # Only show y-label for leftmost plots
                ax.set_ylabel('Run index', fontsize=18, labelpad=10)
            else:
                ax.set_ylabel('')
            
            if row == n_rows - 1:  # Only show x-label for bottom plots
                ax.set_xlabel('Run index', fontsize=18, labelpad=10)
            else:
                ax.set_xlabel('')
            
            # Adjust tick parameters
            ax.tick_params(axis='both', which='major', length=6, width=1.5)
            ax.tick_params(axis='both', which='minor', length=0)
    
    # Hide any unused subplots
    for idx in range(len(m_keys), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)
    
    # Add a horizontal colorbar at the bottom that spans a portion of the width
    cbar_ax = fig.add_subplot(gs[-1, :])
    # Set the position of the colorbar axes to be centered and shorter
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0 + pos.width*0.2,  # Start at 20% from left
                         pos.y0,
                         pos.width*0.6,  # Use 60% of the width
                         pos.height])
    
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='bone_r')
    cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16, length=6, width=1.5)  # Larger colorbar tick labels
    cbar.set_label('Distance', fontsize=18, labelpad=10)  # Larger colorbar label
    
    # Add global title with reduced space to subplots
    fig.suptitle('Preoptimisation subspace distance matrices', 
                 y=1.01, fontsize=22, weight='bold')  # Reduced y and increased font
    
    return fig

 
