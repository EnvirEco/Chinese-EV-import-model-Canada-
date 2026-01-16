"""
Chinese EV Import Impact Model
Estimates market growth and emissions reductions from Chinese EV imports to Canada

Author: Dave Sawyer, Canadian Climate Institute
Date: January 2026
"""

import pandas as pd
import numpy as np

print("="*80)
print("CHINESE EV IMPORT IMPACT MODEL")
print("Market Growth and Emissions Analysis")
print("="*80)

# ============================================================================
# SECTION 1: LOAD ASSUMPTIONS
# ============================================================================

print("\n[1/7] Loading assumptions...")

# Main assumptions
assumptions = pd.read_csv('chinese_ev_assumptions.csv')
params = dict(zip(assumptions['parameter'], assumptions['value']))

# Price segments
segments = pd.read_csv('price_segments.csv')

# Provincial grid data
provincial = pd.read_csv('provincial_grid_data.csv')

print(f"✓ Loaded {len(assumptions)} main parameters")
print(f"✓ Loaded {len(segments)} price segments")
print(f"✓ Loaded {len(provincial)} provincial records")

# Calculate weighted average grid intensity
provincial['penetration_weight'] = provincial['zev_penetration_pct'] / 100
total_penetration = provincial['penetration_weight'].sum()
provincial['normalized_weight'] = provincial['penetration_weight'] / total_penetration

weighted_grid_intensity = (provincial['grid_intensity_g_co2_per_kwh'] * 
                           provincial['normalized_weight']).sum()

print(f"\nWeighted Grid Intensity Calculation:")
print(f"  National average (unweighted): {params['grid_intensity_g_co2_per_kwh']:.0f} g CO2e/kWh")
print(f"  EV sales-weighted average: {weighted_grid_intensity:.0f} g CO2e/kWh")
print(f"  → Using {weighted_grid_intensity:.0f} g/kWh for emissions calculation")

# Override with weighted value
params['grid_intensity_g_co2_per_kwh'] = weighted_grid_intensity

# ============================================================================
# SECTION 2: CALCULATE WEIGHTED CHINESE EV PRICE
# ============================================================================

print("\n[2/7] Calculating weighted Chinese EV prices...")

# Overall weighted average
chinese_weighted_price = (
    params['chinese_low_cost_share'] * params['chinese_low_cost_price'] +
    (1 - params['chinese_low_cost_share']) * params['chinese_standard_price']
)

print(f"\nChinese EV Price Mix:")
print(f"  Low-cost ({params['chinese_low_cost_share']*100:.0f}%): ${params['chinese_low_cost_price']:,.0f}")
print(f"  Standard ({(1-params['chinese_low_cost_share'])*100:.0f}%): ${params['chinese_standard_price']:,.0f}")
print(f"  → Weighted average: ${chinese_weighted_price:,.0f}")

# Calculate weighted price for each segment
segments['chinese_weighted_price'] = (
    params['chinese_low_cost_share'] * segments['chinese_low_price'] +
    (1 - params['chinese_low_cost_share']) * segments['chinese_standard_price']
)

# ============================================================================
# SECTION 3: SEGMENT-LEVEL ANALYSIS (INFORMATIONAL)
# ============================================================================

print("\n[3/7] Analyzing price differentials by segment (informational)...")

# Calculate baseline segment volumes
segments['baseline_volume'] = segments['baseline_market_share'] * params['baseline_market']

# Price differential for each segment
segments['price_differential'] = segments['baseline_canadian_price'] - segments['chinese_weighted_price']
segments['price_differential_pct'] = (segments['price_differential'] / segments['baseline_canadian_price']) * 100

print("\nPrice Competitiveness by Segment:")
print(f"{'Segment':<20} {'Canadian Price':>15} {'Chinese Price':>15} {'Differential':>15}")
print("-" * 70)
for _, row in segments.iterrows():
    print(f"{row['segment']:<20} ${row['baseline_canadian_price']:>14,.0f} ${row['chinese_weighted_price']:>14,.0f} ${row['price_differential']:>14,.0f} ({row['price_differential_pct']:>5.1f}%)")

print("\nNote: Segment analysis shown for context. Main calculation uses market-wide equilibrium.")

# ============================================================================
# SECTION 4: SEGMENT-LEVEL EQUILIBRIUM ITERATION
# ============================================================================

print("\n[4/7] Finding equilibrium market size with segment-level elasticities...")

# Setup baseline values
total_baseline = params['baseline_market']
chinese_volume = params['chinese_quota']
baseline_price = params['baseline_canadian_price']

# Calculate segment baselines
segments['baseline_volume'] = segments['baseline_market_share'] * total_baseline

# Allocate Chinese quota across segments based on price competitiveness
# Chinese weighted avg is $40,340 - most competitive in budget/mid-range
segments['chinese_competitiveness'] = 1 / (segments['baseline_canadian_price'] / segments['chinese_weighted_price'])
total_competitiveness = (segments['chinese_competitiveness'] * segments['baseline_market_share']).sum()
segments['chinese_allocation_share'] = (segments['chinese_competitiveness'] * 
                                        segments['baseline_market_share']) / total_competitiveness
segments['chinese_allocated'] = segments['chinese_allocation_share'] * chinese_volume

print(f"\nChinese EV Allocation by Segment:")
print(f"{'Segment':<20} {'Baseline Vol':>13} {'Chinese Alloc':>14} {'Chinese %':>12}")
print("-" * 65)
for _, row in segments.iterrows():
    print(f"{row['segment']:<20} {row['baseline_volume']:>13,.0f} {row['chinese_allocated']:>14,.0f} {row['chinese_allocation_share']*100:>11.1f}%")

# Initialize segment totals for iteration
segments['segment_total'] = segments['baseline_volume'].copy()

# Equilibrium iteration
max_iterations = 50
convergence_threshold = 100  # EVs
converged = False

print(f"\nStarting iteration (converges when change < {convergence_threshold} EVs):")
print(f"{'Iter':>4} {'Total Market':>15} {'Budget':>12} {'Mid-range':>12} {'Premium':>12} {'Change':>12}")
print("-" * 75)

for iteration in range(max_iterations):
    # Store previous total
    prev_total = segments['segment_total'].sum()
    
    # For each segment, calculate response
    for idx, row in segments.iterrows():
        segment_chinese = row['chinese_allocated']
        segment_baseline = row['baseline_volume']
        segment_canadian_price = row['baseline_canadian_price']
        segment_chinese_price = row['chinese_weighted_price']
        segment_elasticity = row['segment_elasticity']
        
        # Calculate segment weighted average price
        segment_non_chinese = row['segment_total'] - segment_chinese
        if row['segment_total'] > 0:
            segment_avg_price = ((segment_chinese * segment_chinese_price + 
                                 segment_non_chinese * segment_canadian_price) / 
                                row['segment_total'])
        else:
            segment_avg_price = segment_canadian_price
        
        # Calculate price drop and apply segment elasticity
        price_drop = segment_canadian_price - segment_avg_price
        growth_rate = (price_drop / 1000) * segment_elasticity
        
        # New segment volume
        new_segment_volume = segment_baseline * (1 + growth_rate)
        segments.at[idx, 'segment_total'] = new_segment_volume
    
    # Check convergence
    new_total = segments['segment_total'].sum()
    change = abs(new_total - prev_total)
    
    if iteration < 10 or iteration % 5 == 0 or change < convergence_threshold:
        print(f"{iteration+1:4d} {new_total:15,.0f} {segments.loc[0, 'segment_total']:>12,.0f} {segments.loc[1, 'segment_total']:>12,.0f} {segments.loc[2, 'segment_total']:>12,.0f} {change:>12,.0f}")
    
    if change < convergence_threshold:
        converged = True
        print(f"\n✓ CONVERGED after {iteration+1} iterations")
        break

if not converged:
    print(f"\n⚠️  Did not converge after {max_iterations} iterations")

# Calculate final equilibrium values
equilibrium_total = segments['segment_total'].sum()
segments['non_chinese_volume'] = segments['segment_total'] - segments['chinese_allocated']

# Calculate market-wide weighted average price
total_chinese_volume = segments['chinese_allocated'].sum()
total_non_chinese_volume = segments['non_chinese_volume'].sum()
chinese_value = total_chinese_volume * chinese_weighted_price
non_chinese_value = (segments['non_chinese_volume'] * segments['baseline_canadian_price']).sum()
equilibrium_avg_price = (chinese_value + non_chinese_value) / equilibrium_total

print(f"\nSegment-Level Equilibrium:")
print(f"{'Segment':<20} {'Total Vol':>12} {'Chinese':>12} {'Non-Chinese':>12} {'Growth %':>10}")
print("-" * 75)
for _, row in segments.iterrows():
    growth = ((row['segment_total'] - row['baseline_volume']) / row['baseline_volume']) * 100
    print(f"{row['segment']:<20} {row['segment_total']:>12,.0f} {row['chinese_allocated']:>12,.0f} {row['non_chinese_volume']:>12,.0f} {growth:>9.1f}%")

# ============================================================================
# SECTION 5: CALCULATE NET INCREMENTAL EVs
# ============================================================================

print("\n[5/7] Calculating net incremental EV sales...")

net_incremental = equilibrium_total - total_baseline
displacement = total_baseline - total_non_chinese_volume
actual_chinese_volume = total_chinese_volume
non_chinese_volume = total_non_chinese_volume

print(f"\nEquilibrium Market Composition:")
print(f"  Total market: {equilibrium_total:,.0f} EVs")
print(f"  Chinese EVs: {actual_chinese_volume:,.0f} EVs ({actual_chinese_volume/equilibrium_total*100:.1f}%)")
print(f"  Non-Chinese EVs: {non_chinese_volume:,.0f} EVs ({non_chinese_volume/equilibrium_total*100:.1f}%)")
print(f"\nMarket Impact vs. Baseline:")
print(f"  Baseline (2025): {total_baseline:,.0f} EVs @ ${baseline_price:,.0f} avg")
print(f"  With Chinese (2026): {equilibrium_total:,.0f} EVs @ ${equilibrium_avg_price:,.0f} avg")
print(f"  Net incremental EVs: {net_incremental:,.0f} EVs (+{net_incremental/total_baseline*100:.1f}%)")
print(f"  Displacement: {displacement:,.0f} EVs")

# ============================================================================
# SECTION 6: CALCULATE EMISSIONS REDUCTIONS
# ============================================================================

print("\n[6/7] Calculating emissions reductions...")

# Calculate EV charging emissions
annual_ev_consumption = (params['annual_km_driven'] / 100) * params['ev_consumption_kwh_per_100km']
annual_ev_emissions = (annual_ev_consumption * params['grid_intensity_g_co2_per_kwh']) / 1_000_000  # Convert g to tonnes

# Calculate net emissions savings per vehicle
net_emissions_savings_per_vehicle = params['gasoline_vehicle_emissions'] - annual_ev_emissions

print(f"\nEmissions Calculation:")
print(f"  Gasoline vehicle: {params['gasoline_vehicle_emissions']:.2f} tonnes CO2e/year")
print(f"  EV consumption: {annual_ev_consumption:,.0f} kWh/year ({params['ev_consumption_kwh_per_100km']:.1f} kWh/100km × {params['annual_km_driven']:,.0f} km)")
print(f"  Grid intensity: {params['grid_intensity_g_co2_per_kwh']:.0f} g CO2e/kWh (Canadian average)")
print(f"  EV charging emissions: {annual_ev_emissions:.2f} tonnes CO2e/year")
print(f"  → Net savings per vehicle: {net_emissions_savings_per_vehicle:.2f} tonnes CO2e/year")

# Calculate total emissions reductions
annual_emissions_reduction = net_incremental * net_emissions_savings_per_vehicle
lifetime_emissions_reduction = annual_emissions_reduction * params['vehicle_lifetime_years']

print(f"\nTotal Emissions Impact (from {net_incremental:,.0f} net incremental EVs):")
print(f"  Annual reduction: {annual_emissions_reduction:,.0f} tonnes CO2e/year")
print(f"  Lifetime reduction ({params['vehicle_lifetime_years']:.0f} years): {lifetime_emissions_reduction:,.0f} tonnes CO2e")
print(f"  Lifetime reduction: {lifetime_emissions_reduction/1e6:.2f} Mt CO2e")

print(f"\nNote: Uses Canadian average grid intensity ({params['grid_intensity_g_co2_per_kwh']:.0f} g/kWh).")
print(f"      Actual savings likely higher given EV concentration in cleaner-grid provinces.")

# ============================================================================
# SECTION 7: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Summary results
summary = pd.DataFrame({
    'Metric': [
        'Baseline market (2025)',
        'Chinese EV quota',
        'Chinese EV weighted price',
        'Canadian baseline price',
        'Price advantage',
        'Equilibrium total market',
        'Equilibrium avg price',
        'Market growth',
        'Market growth (%)',
        'Chinese EV sales',
        'Chinese market share (%)',
        'Non-Chinese EV sales',
        'Displacement',
        'Net incremental EVs',
        'Weighted grid intensity (g CO2e/kWh)',
        'Net emissions savings per vehicle (tonnes CO2e/year)',
        'Annual emissions reduction (tonnes CO2e)',
        'Lifetime emissions reduction (Mt CO2e)'
    ],
    'Value': [
        f"{total_baseline:,.0f}",
        f"{params['chinese_quota']:,.0f}",
        f"${chinese_weighted_price:,.0f}",
        f"${params['baseline_canadian_price']:,.0f}",
        f"${params['baseline_canadian_price'] - chinese_weighted_price:,.0f} ({((chinese_weighted_price - params['baseline_canadian_price'])/params['baseline_canadian_price'])*100:.1f}%)",
        f"{equilibrium_total:,.0f}",
        f"${equilibrium_avg_price:,.0f}",
        f"{net_incremental:,.0f}",
        f"{(net_incremental/total_baseline)*100:.1f}%",
        f"{actual_chinese_volume:,.0f}",
        f"{(actual_chinese_volume/equilibrium_total)*100:.1f}%",
        f"{non_chinese_volume:,.0f}",
        f"{displacement:,.0f}",
        f"{net_incremental:,.0f}",
        f"{weighted_grid_intensity:.0f}",
        f"{net_emissions_savings_per_vehicle:.2f}",
        f"{annual_emissions_reduction:,.0f}",
        f"{lifetime_emissions_reduction/1e6:.2f}"
    ]
})

summary.to_csv('chinese_ev_results_summary.csv', index=False)
print("✓ Saved: chinese_ev_results_summary.csv")

# Segment-level results
segment_results = segments[[
    'segment',
    'baseline_volume',
    'segment_total',
    'chinese_allocated',
    'non_chinese_volume',
    'segment_elasticity',
    'baseline_canadian_price',
    'chinese_weighted_price'
]].copy()

segment_results['growth'] = segment_results['segment_total'] - segment_results['baseline_volume']
segment_results['growth_pct'] = (segment_results['growth'] / segment_results['baseline_volume']) * 100

segment_results.to_csv('chinese_ev_results_by_segment.csv', index=False)
print("✓ Saved: chinese_ev_results_by_segment.csv")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"  → Baseline (2025): {total_baseline:,.0f} EVs @ ${baseline_price:,.0f} avg")
print(f"  → With Chinese imports: {equilibrium_total:,.0f} EVs @ ${equilibrium_avg_price:,.0f} avg")
print(f"  → Net market growth: {net_incremental:,.0f} EVs (+{(net_incremental/total_baseline)*100:.1f}%)")
print(f"  → Chinese market share: {(actual_chinese_volume/equilibrium_total)*100:.1f}%")
print(f"  → Weighted grid intensity: {weighted_grid_intensity:.0f} g CO2e/kWh (vs. 120 national avg)")
print(f"  → Net emissions per vehicle: {net_emissions_savings_per_vehicle:.2f} tonnes CO2e/year")
print(f"  → Lifetime emissions avoided: {lifetime_emissions_reduction/1e6:.2f} Mt CO2e")
print("\nNote: Used segment-level elasticities and provincial grid weighting")
print("="*80)
