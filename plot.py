warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Plot style ─────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.labelcolor':  '#e0e0e0',
    'xtick.color':      '#a0a0a0',
    'ytick.color':      '#a0a0a0',
    'text.color':       '#e0e0e0',
    'grid.color':       '#2a2d3a',
    'axes.titlecolor':  '#ffffff',
    'figure.titlesize': 16,
})
PALETTE = ['#00d4ff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff8cc8']
MISSING_TOKEN = "__MISSING__"


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — EDA VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════
def run_eda(train: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("  SECTION 1 — EDA & FEATURE VISUALIZATIONS")
    print("="*60)

    df = train.copy()
    df['hour']      = df['timestamp'].apply(lambda t: int(t.split(':')[0]))
    df['minute']    = df['timestamp'].apply(lambda t: int(t.split(':')[1]))
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15

    # ── Plot 1: Demand Distribution ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Traffic Demand Distribution', fontsize=16, fontweight='bold')

    axes[0].hist(df['demand'], bins=60, color=PALETTE[0],
                 edgecolor='none', alpha=0.85)
    axes[0].set_title('Demand Histogram')
    axes[0].set_xlabel('Demand')
    axes[0].set_ylabel('Count')
    axes[0].axvline(df['demand'].mean(), color=PALETTE[1],
                    linestyle='--', linewidth=1.5, label=f"Mean={df['demand'].mean():.3f}")
    axes[0].axvline(df['demand'].median(), color=PALETTE[2],
                    linestyle='--', linewidth=1.5, label=f"Median={df['demand'].median():.3f}")
    axes[0].legend()

    sns.boxplot(x='Weather', y='demand', data=df,
                palette=PALETTE, ax=axes[1])
    axes[1].set_title('Demand by Weather')
    axes[1].set_xlabel('Weather')
    axes[1].set_ylabel('Demand')

    plt.tight_layout()
    plt.savefig('plot1_demand_distribution.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot1_demand_distribution.png")

    # ── Plot 2: Demand by Time ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Demand Patterns Over Time', fontsize=16, fontweight='bold')

    hourly = df.groupby('hour')['demand'].mean().reset_index()
    axes[0].plot(hourly['hour'], hourly['demand'],
                 color=PALETTE[0], linewidth=2.5, marker='o', markersize=4)
    axes[0].fill_between(hourly['hour'], hourly['demand'],
                         alpha=0.2, color=PALETTE[0])
    axes[0].set_title('Average Demand by Hour')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Avg Demand')
    axes[0].axvspan(6, 9,   alpha=0.15, color=PALETTE[3], label='Morning peak')
    axes[0].axvspan(17, 21, alpha=0.15, color=PALETTE[1], label='Evening peak')
    axes[0].legend()

    slot_demand = df.groupby('time_slot')['demand'].mean().reset_index()
    axes[1].plot(slot_demand['time_slot'], slot_demand['demand'],
                 color=PALETTE[2], linewidth=1.5, alpha=0.9)
    axes[1].set_title('Average Demand by 15-min Slot')
    axes[1].set_xlabel('Time Slot (0=00:00, 95=23:45)')
    axes[1].set_ylabel('Avg Demand')

    plt.tight_layout()
    plt.savefig('plot2_demand_time.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot2_demand_time.png")

    # ── Plot 3: Demand by Road & Location Features ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Demand by Road & Location Features', fontsize=16, fontweight='bold')

    sns.boxplot(x='RoadType', y='demand', data=df,
                palette=PALETTE, ax=axes[0])
    axes[0].set_title('Demand by Road Type')

    sns.boxplot(x='LargeVehicles', y='demand', data=df,
                palette=[PALETTE[0], PALETTE[1]], ax=axes[1])
    axes[1].set_title('Demand by Large Vehicles')

    sns.boxplot(x='Landmarks', y='demand', data=df,
                palette=[PALETTE[2], PALETTE[3]], ax=axes[2])
    axes[2].set_title('Demand by Landmarks')

    plt.tight_layout()
    plt.savefig('plot3_demand_road_features.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot3_demand_road_features.png")

    # ── Plot 4: Correlation Heatmap ────────────────────────────────
    num_cols = ['demand', 'NumberofLanes', 'Temperature', 'hour', 'time_slot']
    road_map    = {'Highway': 0, 'Street': 1, 'Residential': 2}
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['road_enc']    = df['RoadType'].map(road_map).fillna(-1)
    df['weather_enc'] = df['Weather'].map(weather_map).fillna(-1)
    df['lv_bin']      = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['lm_bin']      = (df['Landmarks'] == 'Yes').astype(int)

    corr_cols = ['demand', 'NumberofLanes', 'Temperature',
                 'hour', 'time_slot', 'road_enc',
                 'weather_enc', 'lv_bin', 'lm_bin']
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    plt.tight_layout()
    plt.savefig('plot4_correlation_heatmap.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot4_correlation_heatmap.png")

    # ── Plot 5: Top geohash demand ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Geospatial Demand Patterns', fontsize=16, fontweight='bold')

    top_geo = df.groupby('geohash')['demand'].mean().nlargest(20)
    axes[0].barh(range(len(top_geo)), top_geo.values,
                 color=PALETTE[0], alpha=0.85)
    axes[0].set_yticks(range(len(top_geo)))
    axes[0].set_yticklabels(top_geo.index, fontsize=8)
    axes[0].set_title('Top 20 Geohashes by Avg Demand')
    axes[0].set_xlabel('Avg Demand')

    demand_by_lanes = df.groupby('NumberofLanes')['demand'].mean().reset_index()
    axes[1].bar(demand_by_lanes['NumberofLanes'].astype(str),
                demand_by_lanes['demand'],
                color=PALETTE[2], alpha=0.85)
    axes[1].set_title('Avg Demand by Number of Lanes')
    axes[1].set_xlabel('Number of Lanes')
    axes[1].set_ylabel('Avg Demand')

    plt.tight_layout()
    plt.savefig('plot5_geo_demand.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot5_geo_demand.png")

    # ── Plot 6: Temperature vs Demand ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Temperature & Weather vs Demand', fontsize=16, fontweight='bold')

    sample = df.dropna(subset=['Temperature']).sample(
        min(3000, len(df)), random_state=42)
    axes[0].scatter(sample['Temperature'], sample['demand'],
                    c=PALETTE[0], alpha=0.3, s=8)
    axes[0].set_title('Temperature vs Demand')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Demand')

    weather_hour = df.groupby(['Weather', 'hour'])['demand'].mean().reset_index()
    for i, weather in enumerate(df['Weather'].dropna().unique()):
        sub = weather_hour[weather_hour['Weather'] == weather]
        axes[1].plot(sub['hour'], sub['demand'],
                     label=weather, color=PALETTE[i % len(PALETTE)],
                     linewidth=2, marker='o', markersize=3)
    axes[1].set_title('Hourly Demand by Weather')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('Avg Demand')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('plot6_temp_weather_demand.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  ✅ plot6_temp_weather_demand.png")

    print("\n  All plots saved in current directory.")