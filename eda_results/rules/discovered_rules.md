# Discovered LTN Rule Candidates

Total rules discovered: 16

## Mathematical Rules (6)

### MATH_001: Dry_Total_g = Dry_Clover_g + Dry_Dead_g + Dry_Green_g

**Pseudo-Logic**: `FORALL x: Dry_Total_g(x) ≈ Dry_Clover_g(x) + Dry_Dead_g(x) + Dry_Green_g(x)`

**Constraint Type**: equality

**LTN Encoding**: soft_constraint

**Confidence**: high

**Priority**: 1

**Validation Metrics**:
  - mean_residual: 0.0008784313725502937
  - max_residual: 0.30879999999999974
  - std_residual: 0.016343440616208276
  - violation_rate: 0.0

---

### MATH_004: GDM_g = 0.799 × Dry_Total_g + -2.925

**Pseudo-Logic**: `FORALL x: GDM_g(x) ≈ 0.799 * Dry_Total_g(x) + -2.925`

**Constraint Type**: linear_relationship

**LTN Encoding**: soft_constraint

**Confidence**: high

**Priority**: 2

**Validation Metrics**:
  - r_squared: 0.803607125816123
  - p_value: 1.59778273987666e-127
  - slope: 0.7987953948986295
  - intercept: -2.925473175038846

---

### MATH_003: GDM_g = 0.868 × Dry_Green_g + 10.163

**Pseudo-Logic**: `FORALL x: GDM_g(x) ≈ 0.868 * Dry_Green_g(x) + 10.163`

**Constraint Type**: linear_relationship

**LTN Encoding**: soft_constraint

**Confidence**: high

**Priority**: 2

**Validation Metrics**:
  - r_squared: 0.7819101969245841
  - p_value: 1.9381144268178882e-119
  - slope: 0.8680551853868375
  - intercept: 10.162685378904225

---

### MATH_002: Dry_Total_g = 0.915 × Dry_Green_g + 20.963

**Pseudo-Logic**: `FORALL x: Dry_Total_g(x) ≈ 0.915 * Dry_Green_g(x) + 20.963`

**Constraint Type**: linear_relationship

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 2

**Validation Metrics**:
  - r_squared: 0.6894222862012327
  - p_value: 3.689413362977551e-92
  - slope: 0.9147404698423762
  - intercept: 20.963385552857265

---

### MATH_010: GDM strongly correlates with green biomass (R²=0.782)

**Pseudo-Logic**: `FORALL x: Dry_Green_g(x) ≈ 0.901 * GDM_g(x) + -3.348`

**Constraint Type**: correlation

**LTN Encoding**: soft_constraint

**Confidence**: high

**Priority**: 2

**Validation Metrics**:
  - r_squared: 0.7819101969245841
  - p_value: 1.9381144268178882e-119
  - correlation: 0.8842568613952533

---

### MATH_011: GDM correlates with total biomass (R²=0.804)

**Pseudo-Logic**: `FORALL x: Dry_Total_g(x) ≈ 1.006 * GDM_g(x) + 11.843`

**Constraint Type**: correlation

**LTN Encoding**: soft_constraint

**Confidence**: high

**Priority**: 2

**Validation Metrics**:
  - r_squared: 0.803607125816123
  - p_value: 1.59778273987666e-127
  - correlation: 0.8964413677514682

---

## Species Rules (4)

### SPECIES_020: Species 'SubcloverLosa' has high clover proportion (mean=1.00)

**Pseudo-Logic**: `FORALL x: Species(x) = 'SubcloverLosa' → Dry_Clover_g(x) / Dry_Total_g(x) ≥ 0.19`

**Constraint Type**: threshold

**LTN Encoding**: soft_constraint

**Confidence**: high

**Priority**: 2

**Validation Metrics**:
  - mean_proportion: 1.0
  - threshold: 0.19034694131299462
  - sample_count: 5

---

### SPECIES_021: Species 'Clover' has high clover proportion (mean=0.70)

**Pseudo-Logic**: `FORALL x: Species(x) = 'Clover' → Dry_Clover_g(x) / Dry_Total_g(x) ≥ 0.19`

**Constraint Type**: threshold

**LTN Encoding**: soft_constraint

**Confidence**: high

**Priority**: 2

**Validation Metrics**:
  - mean_proportion: 0.7028377094638941
  - threshold: 0.19034694131299462
  - sample_count: 41

---

### SPECIES_022: Species 'WhiteClover' has high clover proportion (mean=0.32)

**Pseudo-Logic**: `FORALL x: Species(x) = 'WhiteClover' → Dry_Clover_g(x) / Dry_Total_g(x) ≥ 0.19`

**Constraint Type**: threshold

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 2

**Validation Metrics**:
  - mean_proportion: 0.3216211780668726
  - threshold: 0.19034694131299462
  - sample_count: 10

---

### SPECIES_023: Species 'Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass' has high clover proportion (mean=0.19)

**Pseudo-Logic**: `FORALL x: Species(x) = 'Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass' → Dry_Clover_g(x) / Dry_Total_g(x) ≥ 0.19`

**Constraint Type**: threshold

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 2

**Validation Metrics**:
  - mean_proportion: 0.19034694131299462
  - threshold: 0.19034694131299462
  - sample_count: 7

---

## Temporal Rules (2)

### TEMPORAL_030: Dead biomass proportion peaks in Winter

**Pseudo-Logic**: `FORALL x: season(x) = 'Winter' → Dry_Dead_g(x) / Dry_Total_g(x) ≥ 0.240`

**Constraint Type**: seasonal_threshold

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 3

**Validation Metrics**:
  - season: Winter
  - mean_proportion: 0.3005903778172893
  - pattern: dead_maximization

---

### TEMPORAL_031: Green biomass proportion peaks in Summer

**Pseudo-Logic**: `FORALL x: season(x) = 'Summer' → Dry_Green_g(x) / Dry_Total_g(x) ≥ 0.657`

**Constraint Type**: seasonal_threshold

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 3

**Validation Metrics**:
  - season: Summer
  - mean_proportion: 0.8208324989669021
  - pattern: green_maximization

---

## Geographic Rules (4)

### GEO_040: State NSW has NDVI in range [0.34, 0.89]

**Pseudo-Logic**: `FORALL x: State(x) = 'NSW' → 0.340 ≤ NDVI(x) ≤ 0.890`

**Constraint Type**: range_constraint

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 3

**Validation Metrics**:
  - state: NSW
  - mean_ndvi: 0.6564
  - min_ndvi: 0.34
  - max_ndvi: 0.89

---

### GEO_041: State Tas has NDVI in range [0.19, 0.91]

**Pseudo-Logic**: `FORALL x: State(x) = 'Tas' → 0.190 ≤ NDVI(x) ≤ 0.910`

**Constraint Type**: range_constraint

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 3

**Validation Metrics**:
  - state: Tas
  - mean_ndvi: 0.6273188405797101
  - min_ndvi: 0.19
  - max_ndvi: 0.91

---

### GEO_042: State Vic has NDVI in range [0.30, 0.87]

**Pseudo-Logic**: `FORALL x: State(x) = 'Vic' → 0.300 ≤ NDVI(x) ≤ 0.870`

**Constraint Type**: range_constraint

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 3

**Validation Metrics**:
  - state: Vic
  - mean_ndvi: 0.7126785714285714
  - min_ndvi: 0.3
  - max_ndvi: 0.87

---

### GEO_043: State WA has NDVI in range [0.16, 0.81]

**Pseudo-Logic**: `FORALL x: State(x) = 'WA' → 0.160 ≤ NDVI(x) ≤ 0.810`

**Constraint Type**: range_constraint

**LTN Encoding**: soft_constraint

**Confidence**: medium

**Priority**: 3

**Validation Metrics**:
  - state: WA
  - mean_ndvi: 0.5962500000000001
  - min_ndvi: 0.16
  - max_ndvi: 0.81

---

