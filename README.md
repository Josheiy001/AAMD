Dataset link
https://archive.ics.uci.edu/dataset/45/heart+disease

Objective:
To develop and interpret a general cardiovascular disease (CVD) risk model using UCI data, integrating multivariate exploration (PCA, Factor Analysis), phenotype discovery (clustering), and prediction (penalized logistic regression) with rigorous calibration and clinical-utility assessment.
Data: UCI Heart Disease (Cleveland; ~303 observations, 13 predictors) with a binary CVD outcome; Heart Failure Clinical Records (~299) used only to probe robustness of latent structure.
Proposal: 
After standardized preprocessing and stratified splitting, we applied PCA and maximum-likelihood Factor Analysis with varimax rotation to uncover latent CVD dimensions from continuous variables (age, blood pressure, cholesterol, heart-rate response, ST depression). We derived patient phenotypes by clustering factor scores combined with categorical ischemia markers. Multiple linear regression related risk factors to ST depression (oldpeak) as a continuous ischemia surrogate.

PCA notes (for Klei)
1. Select best Variable set for PCA/FA:
Use continuous risk variables only (e.g., age, trestbps, chol, thalach, oldpeak; optionally ca if treated as numeric).
Why: Categorical variables distort PCA/FA on Pearson correlations.

2. Scale:
Use z-scores (mean 0, SD 1). Units differ; PCA/FA are scale-sensitive.

3. Decide on How many components/factors:
Use Parallel Analysis as the primary rule; sanity-check with scree elbow and interpretability (expect 2–3).
Why: PA outperforms ad-hoc rules.

4. Rotation (FA):
Start varimax; if factors clearly correlate, report oblimin as a sensitivity.
Why: Rotation clarifies factor meaning.

5. Retention threshold:
Treat |loading| ≥ 0.35 as salient when labeling components/factors.
Why: Prevents over-interpreting noise.

6. Output to downstream:
Provide scores (per subject) for retained PCs/factors for clustering and regression.

Outcomes: 
PCA
Scree plot (eigenvalues).
Parallel Analysis plot (empirical vs random eigenvalues) + statement “retain k”.
Loadings table (variables × PCs) + a heatmap; cumulative explained variance table.
PC scores (k columns) saved for others.

FA
KMO value and Bartlett p-value with one-line interpretation (e.g., “KMO=0.71, adequate; Bartlett p<.001, correlations not spherical”).
Rotated loading matrix (varimax; oblimin optional) with salient loadings highlighted.
Communalities & uniqueness table (flag any communality <0.30).
Factor scores (k columns) saved for others.

4) How to interpret (and what to write)

Name each PC/Factor by the strongest loadings:
Example: “Metabolic/Pressure” if age, trestbps, chol load together.
Example: “Ischemia/Exertion” if oldpeak (+) and thalach (−) dominate.

Explain variance coverage: “PC1+PC2 explain ~X% of variance; factors capture shared structure relevant to hemodynamics and ischemic response.”



MRA notes (for Johannes)
What is OLDPEAK?

In the Cleveland Heart Disease data, oldpeak = ST depression induced by exercise relative to rest (in mm).

Larger values generally indicate more myocardial ischemia during stress (poorer perfusion → ST segment drops below baseline).

It’s continuous, not just yes/no—so it naturally quantifies how much ischemia is evident, i.e., an ischemia burden.

Why run a Multiple Regression on OLDPEAK?

Your course requires a Multiple Regression Analysis (MRA). Modeling oldpeak (continuous) against risk factors:

Satisfies the “multiple regression” requirement with a clinically meaningful target.

Complements your main CVD classification model by explaining mechanisms: which risk factors drive ischemic response under stress?

Produces interpretable effect sizes (per-unit change in predictors → mm change in ST depression).

Model setup (conceptual)

Outcome (Y): oldpeak (continuous, mm)
Predictors (X): pick from age, sex, trestbps, chol, thalach, fbs, cp, restecg, exang, slope, thal, ca (after proper encoding).


Encode categoricals (e.g., cp, restecg, slope, thal) as dummies; choose a clinically sensible reference (e.g., cp=typical angina, slope=upsloping).

Consider interactions you think are mechanistic:

exang × slope (exercise-induced angina tends to pair with down-sloping ST)

age × sex (sex differences across age)

thal × ca (perfusion defect plus coronary vessel count)

