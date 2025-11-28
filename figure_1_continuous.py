import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def replicate_figure1(rate_file, stim_file, sampling_period_ms=8, 
                     response_delay_ms=30, duration_s=6):
    """
    Replicate Figure 1 from Brenner et al. 2000
    
    Panel (a): r(t) and s(t) on same plot with aligned zeros
    Panel (b): r(s) - response as function of velocity 30ms earlier
    """
    
    print("="*60)
    print("REPLICATING FIGURE 1")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    rate_data = pd.read_csv(rate_file)
    rate_times = rate_data['time_s'].values
    rate = rate_data['rate_hz'].values  # change column name as needed
    
    stim = pd.read_csv(stim_file, header=None).values.flatten()
    stim_times = np.arange(len(stim)) * sampling_period_ms / 1000
    
    print(f"  Rate: {len(rate)} points")
    print(f"  Stimulus: {len(stim)} samples")
    
    # ========== PANEL A: r(t) and s(t) ==========
    print("\nCreating Panel (a): r(t) and s(t)...")
    
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Find samples for the time window
    rate_mask = rate_times < duration_s
    stim_mask = stim_times < duration_s
    
    rate_in_window = rate[rate_mask]
    stim_in_window = stim[stim_mask]
    
    # Plot firing rate as continuous line (left axis)
    ax1 = ax_a
    ax1.plot(rate_times[rate_mask], rate_in_window, '-', 
             linewidth=1.5, color='blue', label='Firing rate r(t)')
    ax1.set_ylabel('Firing Rate (spikes/s)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlim([0, duration_s])
    
    # Plot stimulus (right axis)
    ax2 = ax1.twinx()
    ax2.plot(stim_times[stim_mask], stim_in_window, '-', 
             color='gray', linewidth=1.5, label='Velocity s(t)')
    ax2.set_ylabel('Velocity (°/s)', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Align zeros properly
    # Get data ranges
    rate_max = np.max(rate_in_window)
    stim_min = np.min(stim_in_window)
    stim_max = np.max(stim_in_window)
    
    # Calculate where zero should be as a fraction of the axis
    # For stimulus, zero is at: -stim_min / (stim_max - stim_min)
    stim_zero_fraction = -stim_min / (stim_max - stim_min)
    
    # Now set rate axis so zero is at the same fraction
    # If zero is at fraction f: (0 - rate_min) / (rate_max - rate_min) = f
    # Solving: rate_min = -f * (rate_max - rate_min)
    # rate_min = -f * rate_max + f * rate_min
    # rate_min * (1 - f) = -f * rate_max
    # rate_min = -f * rate_max / (1 - f)
    
    rate_min = -stim_zero_fraction * rate_max / (1 - stim_zero_fraction)
    
    # Set limits with small padding
    ax1.set_ylim([rate_min, rate_max * 1.05])
    ax2.set_ylim([stim_min * 1.05, stim_max * 1.05])
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_title('(a) Response to Slowly Varying Inputs', 
                  fontsize=13, fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line at zero for both axes
    ax1.axhline(0, color='blue', linestyle=':', alpha=0.5, linewidth=0.8)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    # ========== PANEL B: r(s) ==========
    print("Creating Panel (b): r(s)...")
    
    # Align rate and stimulus with delay
    rate_aligned = []
    stim_aligned = []
    
    for i, t in enumerate(rate_times):
        # Time 30ms before this rate measurement
        stim_time = t - (response_delay_ms / 1000)
        
        if stim_time >= 0 and stim_time < stim_times[-1]:
            # Find closest stimulus sample
            stim_idx = np.argmin(np.abs(stim_times - stim_time))
            stim_aligned.append(stim[stim_idx])
            rate_aligned.append(rate[i])
    
    stim_aligned = np.array(stim_aligned)
    rate_aligned = np.array(rate_aligned)
    
    print(f"  Aligned {len(rate_aligned)} rate-stimulus pairs")
    
    # Scatter plot
    ax_b.plot(stim_aligned, rate_aligned, 'o', 
              markersize=2, alpha=0.3, color='gray',
              markerfacecolor='none', label='Raw data')
    
    # Bin and average for smooth curve
    n_bins = 30
    velocity_bins = np.linspace(stim_aligned.min(), stim_aligned.max(), n_bins + 1)
    rate_binned = []
    velocity_centers = []
    
    for i in range(n_bins):
        mask = (stim_aligned >= velocity_bins[i]) & (stim_aligned < velocity_bins[i+1])
        if np.sum(mask) > 10:  # Only include bins with enough data
            rate_binned.append(np.mean(rate_aligned[mask]))
            velocity_centers.append((velocity_bins[i] + velocity_bins[i+1]) / 2)
    
    # Plot averaged curve
    ax_b.plot(velocity_centers, rate_binned, 'o-', 
              markersize=6, color='black', linewidth=2,
              markerfacecolor='black', label='Binned average')
    
    ax_b.set_xlabel('Velocity (°/s)', fontsize=12)
    ax_b.set_ylabel('Firing Rate (spikes/s)', fontsize=12)
    ax_b.set_title('(b) Input-Output Relation r(s)', 
                   fontsize=13, fontweight='bold', loc='left')
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=10)
    
    # Add vertical and horizontal lines at zero
    ax_b.axhline(0, color='black', linestyle=':', alpha=0.3, linewidth=0.8)
    ax_b.axvline(0, color='black', linestyle=':', alpha=0.3, linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('figure1_replication.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Figure 1 replication complete!")
    print("  Saved to: figure1_replication.png")
    
    return {
        'rate': rate,
        'rate_times': rate_times,
        'stim': stim,
        'stim_aligned': stim_aligned,
        'rate_aligned': rate_aligned,
        'velocity_centers': velocity_centers,
        'rate_binned': rate_binned
    }


# ========== REPLICATE FIGURE 1 FOR STIMULUS 1 ==========

print("\n### STIMULUS 1 (σ = 46) ###\n")
result1 = replicate_figure1(
    rate_file='rate_avg_46.csv',
    stim_file='slow_stim_46(3).csv',
    sampling_period_ms=8,
    response_delay_ms=30,
    duration_s=6  # how much time to include
)

# continuous in panel b too?