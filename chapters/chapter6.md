# Chapter 6: Evolutionary Conservation and Traditional Tools

**[Interactive: Chapter 6](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter6.html)**

## Opening Vignette: The Needle in a Haystack Problem

Dr. Sarah Chen stares at her screen, looking at whole-genome sequencing results from a 4-year-old patient with severe developmental delays and seizures. The sequencing worked perfectly—excellent coverage, high-quality reads. But now comes the overwhelming part.

The analysis pipeline identified **4.2 million genetic variants** in this child's genome compared to the reference. That's 4.2 million differences. Somewhere in those millions of variants, there might be one—or maybe a handful—that explain why this child can't walk, can't speak, and has daily seizures.

She filters by frequency first. Removing common variants (those found in more than 1% of the population) drops the count to about **50,000 rare variants**. Better, but still impossible to work with. She then focuses on protein-coding regions, leaving her with roughly **2,500 rare variants** that could potentially change protein sequences.

2,500 variants. Each one could theoretically affect a protein. But which ones actually matter? Which changes are harmless quirks of individual variation, and which ones break something critical?

Here's what Dr. Chen can't do: she can't test all 2,500 variants experimentally. Each functional test—expressing the mutant protein, measuring its activity, testing it in cells or model organisms—takes weeks to months and costs thousands of dollars. Testing 2,500 variants would take decades and millions of dollars. The patient needs answers now, not in 2050.

This is where **evolutionary conservation** becomes her most powerful tool.

---

## The Biological Challenge: Why We Need Conservation-Based Prediction

The scale of modern genomics creates a fundamental problem: **we can sequence genomes faster than we can understand them**. A single whole-genome sequence generates millions of variants. Even after filtering by frequency and focusing on coding regions, we're left with thousands of candidate variants that could potentially cause disorders.

We can't functionally test every variant. Here's why:

**Time constraints**: Functional experiments take weeks to months per variant. A patient with an undiagnosed disorder needs answers in weeks or months, not years.

**Cost barriers**: Each functional assay costs $2,000-10,000 depending on complexity. Testing thousands of variants would cost millions of dollars per patient—completely impractical for clinical diagnostics.

**Technical limitations**: Many human proteins can't be easily studied in cell culture or model organisms. Some require specific developmental contexts, cell types, or tissue environments that are difficult or impossible to recreate experimentally.

**Phenotype complexity**: Even when we can test a variant, interpreting the results isn't straightforward. A protein might retain 80% of its normal function—is that enough? Does partial loss of function cause a disorder? The answer depends on the gene and the biological context.

But here's the key insight that makes variant prioritization possible: **evolution has already done millions of experiments for us**.

Every species alive today represents a lineage that has survived millions of years of natural selection. Mutations that break critical biological functions tend to disappear because organisms carrying them don't survive to reproduce. Meanwhile, mutations in less important regions—or neutral changes that don't affect function—accumulate freely over time.

This means that if we compare a DNA sequence across many species and find that a particular nucleotide is identical (or nearly identical) in humans, chimpanzees, mice, dogs, chickens, and even fish, that nucleotide is probably doing something important. Any mutation at that position likely has functional consequences.

Conversely, if a nucleotide varies freely across species, mutations there probably don't matter much.

This is the foundation of **conservation-based variant prediction**: we use evolutionary patterns—the signature of natural selection acting over millions of years—to predict which variants are likely to have functional impact without doing direct experiments.

In this chapter, you'll learn:
- How conservation is measured across species (PhyloP, PhastCons, GERP++)
- How conservation helps predict variant effects in proteins (SIFT, PolyPhen-2)
- How to use these tools to prioritize variants in real diagnostic workflows
- The strengths and limitations of conservation-based prediction

By the end, you'll understand how Dr. Chen can narrow down 2,500 candidates to perhaps 10-20 high-priority variants that warrant further investigation—making the impossible problem solvable.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- [ ] Explain why evolutionary conservation indicates functional importance
- [ ] Interpret conservation scores (PhyloP, PhastCons, GERP++) for DNA sequences
- [ ] Use SIFT and PolyPhen-2 to predict the impact of amino acid changes
- [ ] Describe how multiple sequence alignments reveal conserved positions
- [ ] Apply conservation-based filtering to prioritize variants in a diagnostic setting
- [ ] Recognize the limitations of conservation-based predictions
- [ ] Understand when conservation-based tools work well and when they fail

---

## 6.1 The Logic of Conservation

Before we dive into specific tools and scores, let's make sure the underlying logic is crystal clear.

### Why Conservation Matters

Imagine you're an archaeologist studying ancient texts. You find ten copies of the same manuscript from different time periods, written by scribes who copied from earlier versions. Most words vary between copies—the scribes made mistakes, used different spelling, or updated old language. But a few specific phrases appear **identically** in all ten manuscripts.

What does that tell you? Those identical phrases are probably important—maybe they're religious invocations, legal terms, or critical instructions that scribes were trained never to alter. The consistency across copies, despite errors elsewhere, signals importance.

Evolution works similarly, but instead of manuscripts, we're comparing DNA sequences across species. Instead of scribes, we have mutation and natural selection. And instead of centuries, we're looking across millions of years.

### How Evolution Acts as a Filter

DNA mutates constantly. In humans, each newborn carries about 70 new mutations that neither parent had. Multiply that across millions of individuals and billions of years, and you get an enormous amount of genetic variation.

But not all mutations persist. Natural selection acts as a filter:

**For critical sequences**: Mutations that break important functions reduce survival or reproduction. Organisms carrying these mutations are less likely to pass them on. Over time, such mutations disappear from the population. The result? These sequences stay nearly identical across millions of years and across many species.

**For non-critical sequences**: Mutations that don't affect function—or affect it only mildly—accumulate freely. These sequences drift and diverge between species. After millions of years, they might be completely different.

This creates a clear pattern: **high conservation = functional importance**.

### A Concrete Example: Insulin

Let's look at a real gene: *INS*, which encodes insulin. Insulin is critical for regulating blood sugar. Without functional insulin, you get diabetes.

If we compare the insulin protein sequence across species, we find:
- Humans and chimpanzees: 100% identical (0 differences)
- Humans and mice: 83% identical
- Humans and chickens: 73% identical  
- Humans and zebrafish: 56% identical

Even after 450 million years of evolution separating humans from fish, more than half of insulin's sequence remains unchanged. Why? Because mutations in insulin are often harmful—they cause diabetes or are lethal during development. Natural selection removes them.

Now compare this to a less critical gene—say, one involved in hair color or scent detection. Those genes vary much more freely between species because mutations there don't usually affect survival.

### What Conservation Tells Us About Variants

When we find a new variant in a patient, we can ask: **Does this variant occur at a highly conserved position?**

**Variant at a conserved position**: This nucleotide has been preserved for millions of years. Changing it likely disrupts function. The variant is more likely to have functional impact.

**Variant at a non-conserved position**: This nucleotide varies naturally between species. Changing it is probably harmless. The variant is less likely to cause problems.

This doesn't give us absolute certainty—conservation is a probability, not a guarantee—but it dramatically narrows our search space.

### Conservation Works at Multiple Levels

We can measure conservation at different scales:

**Nucleotide level**: Is this specific DNA base conserved across species?
- Tools: PhyloP, PhastCons, GERP++
- Useful for: Any position in the genome (coding or non-coding)

**Amino acid level**: Is this position in the protein sequence conserved?
- Tools: SIFT, PolyPhen-2
- Useful for: Missense variants (amino acid changes)

**Structural level**: Does the 3D structure of the protein remain similar?
- Tools: Structure-based predictions (we'll cover these in later chapters)
- Useful for: Understanding why certain positions matter

In the next sections, we'll explore how each of these levels of conservation is measured and applied to variant interpretation.

---

## 6.2 Measuring DNA Conservation Across Species

Now let's get specific: how do we actually measure whether a DNA position is conserved?

The process involves three steps:
1. **Align sequences** from many species
2. **Calculate conservation scores** based on evolutionary models
3. **Interpret scores** to predict variant impact

### Step 1: Multiple Sequence Alignment

First, we need to compare the same genomic region across many species. This requires a **multiple sequence alignment (MSA)**—lining up DNA sequences so we can see which positions match and which differ.

Here's a simple example. Let's say we're looking at a short stretch of DNA around a specific position:

```
Human:     ATGCGATCGAGTC
Chimp:     ATGCGATCGAGTC
Mouse:     ATGCGCTCGAGTC
Dog:       ATGCGATCGAGTC
Chicken:   ATGCCATCGAGTC
Zebrafish: ATGCCATAGAGTC
           ||||  || ||||
           Conservation pattern
```

At position 6, we see:
- Human, chimp, dog: **A**
- Mouse: **C**  
- Chicken, zebrafish: **A**

Most species have **A**, but mouse has **C**. This position is **highly conserved**, meaning mutations here are rare and probably disrupt function.

Now look at position 5:
- Humans, chimps, dogs: **G**
- Mouse: **G**
- Chicken: **C**
- Zebrafish: **C**

This position varies more—mammals have G, but birds and fish have C. This suggests the position is **less conserved** and more tolerant of variation.

Real alignments use dozens to hundreds of species and sophisticated algorithms to handle insertions, deletions, and complex evolutionary relationships. Projects like the **Zoonomia Project** (241 mammal genomes) and the **Vertebrate Genomes Project** provide the data for these alignments.

### Step 2: Conservation Scores

Once we have an alignment, we calculate conservation scores. Several different scoring methods exist, each with slightly different approaches:

#### PhyloP (Phylogenetic P-value)

**What it measures**: PhyloP asks, "Is this position evolving slower or faster than expected by chance?"

**How it works**: PhyloP uses a statistical model of neutral evolution (no selection). It compares the observed rate of change at each position to what we'd expect under neutrality.

**Score interpretation**:
- **Positive scores**: Position evolves slower than expected → conserved → likely functional
- **Negative scores**: Position evolves faster than expected → not conserved → likely neutral
- **Score around zero**: Evolution matches neutral expectation

**Typical ranges**:
- PhyloP > 2: Strong conservation (top ~5% of genome)
- PhyloP > 5: Very strong conservation (top ~0.1%)
- PhyloP < -2: Fast evolution (possibly under positive selection)

**Example**: A variant at position chr7:117,548,401 has a PhyloP score of 6.2. This means this position is in the top 0.1% most conserved positions in the genome. A mutation here is very likely to have functional consequences.

#### PhastCons (Phylogenetic Analysis with Space/Time models)

**What it measures**: PhastCons identifies **conserved elements**—stretches of DNA (not just single positions) that are conserved as a block.

**How it works**: Instead of scoring each position independently, PhastCons looks for regions where many adjacent positions are all conserved. It uses a hidden Markov model to segment the genome into conserved and non-conserved elements.

**Score interpretation**:
- **Score 0 to 1**: Probability that a position belongs to a conserved element
- PhastCons > 0.8: High probability of being in a conserved element
- PhastCons < 0.2: Probably not conserved

**Why it's useful**: PhastCons is better at identifying regulatory elements (enhancers, promoters) because these often consist of multiple conserved transcription factor binding sites clustered together.

**Example**: A variant falls in a region with PhastCons score 0.95, meaning there's a 95% probability this region is under selective constraint. Even if the specific nucleotide isn't perfectly conserved, the region as a whole matters.

#### GERP++ (Genomic Evolutionary Rate Profiling)

**What it measures**: GERP++ calculates "rejected substitutions"—how many mutations evolution has rejected at this position compared to neutral expectation.

**How it works**:
1. Calculate expected number of substitutions at a position under neutral evolution
2. Count observed substitutions in the alignment
3. Subtract: GERP score = Expected - Observed

**Score interpretation**:
- **Positive GERP**: Fewer substitutions than expected → conserved
- **Negative GERP**: More substitutions than expected → not conserved
- GERP > 2: Moderate conservation
- GERP > 4: Strong conservation
- GERP > 6: Very strong conservation

**Why it's useful**: GERP scores are intuitive—the score literally tells you how many mutations have been "rejected" by natural selection.

**Example**: A position has GERP++ score of 5.8. This means ~6 mutations that should have occurred by chance were removed by selection. This position is critical.

### Comparing the Three Scores

All three scores measure conservation, but with different approaches:

| Score | Measures | Best For | Interpretation |
|-------|----------|----------|----------------|
| **PhyloP** | Single-nucleotide conservation | Prioritizing individual variants | Positive = conserved |
| **PhastCons** | Conserved elements/regions | Finding regulatory regions | 0-1 probability scale |
| **GERP++** | Rejected substitutions | Understanding selection strength | Number of rejected mutations |

In practice, you'll often see all three scores reported for a variant. They usually agree—if one says "conserved," the others do too—but occasionally they differ, which can provide nuance.

### Practical Application: Filtering Variants

Here's how Dr. Chen might use conservation scores to prioritize her 2,500 rare coding variants:

**Step 1**: Filter by conservation
- Keep variants with PhyloP > 2 OR GERP > 4
- This removes variants at non-conserved positions
- Reduces list from 2,500 to ~400 variants

**Step 2**: Combine with other information
- Check if variant changes amino acid (missense)
- Check if variant creates a stop codon (nonsense)
- Check variant frequency in control populations

**Step 3**: Focus on top candidates
- Variants at highly conserved positions (PhyloP > 5) get highest priority
- Variants creating protein-truncating changes get high priority
- Variants in genes associated with the patient's phenotype get high priority

This workflow doesn't give definitive answers, but it makes the problem tractable—narrowing 2,500 candidates to maybe 20-30 that warrant detailed investigation.

---

## 6.3 Predicting Protein-Level Effects: SIFT and PolyPhen-2

Conservation scores like PhyloP and GERP work at the DNA level—they tell us if a nucleotide position is conserved. But for **missense variants** (variants that change one amino acid to another), we want to know specifically: **Does this amino acid change disrupt protein function?**

This requires protein-level analysis. The two most widely used tools are **SIFT** and **PolyPhen-2**.

### Understanding Missense Variants

A missense variant changes one amino acid in a protein to a different amino acid. For example:

```
Normal:  ...Leucine - Proline - Glycine - Alanine...
Variant: ...Leucine - Proline - Arginine - Alanine...
                                  ^
                         Glycine → Arginine
```

Whether this change matters depends on:
1. **Location**: Is this amino acid in a critical region (active site, binding pocket)?
2. **Chemical properties**: How different are the two amino acids?
3. **Conservation**: Is this position highly conserved across species?
4. **Structural impact**: Does the change disrupt the protein's 3D structure?

SIFT and PolyPhen-2 integrate multiple sources of information to predict whether an amino acid substitution will be tolerated or damaging.

### SIFT (Sorting Intolerant From Tolerant)

**Core principle**: SIFT uses evolutionary conservation of amino acids across species. If an amino acid position has remained the same across many species, changing it is probably harmful.

**How it works**:

1. **Build alignment**: Collect protein sequences from many species (orthologs—the same protein in different species)

2. **Calculate position-specific profile**: For each amino acid position, count which amino acids appear across species

3. **Score the substitution**: If the variant introduces an amino acid rarely or never seen at that position in other species, it's probably damaging

**Score interpretation**:
- **SIFT score**: 0 to 1
- SIFT < 0.05: Predicted "deleterious" (damaging)
- SIFT ≥ 0.05: Predicted "tolerated"

**Example**: Position 456 in protein BRCA1
- Normally: Glycine (G)
- Across 100 species: 98 have Glycine, 2 have Alanine
- Variant changes Glycine → Arginine

SIFT score = 0.01 (very low) → "Deleterious"

Why? Arginine never appears at this position in any species. The substitution is unprecedented in evolution, suggesting it will break function.

**Strengths**:
- Simple, interpretable
- Fast computation
- Works well for evolutionarily conserved proteins

**Limitations**:
- Requires orthologs (doesn't work for human-specific genes)
- Doesn't consider protein structure
- Binary classification can oversimplify

### PolyPhen-2 (Polymorphism Phenotyping v2)

**Core principle**: PolyPhen-2 integrates multiple features—conservation, structural information, and physicochemical properties—using a machine learning model to predict variant impact.

**How it works**:

1. **Conservation analysis**: Like SIFT, checks if substitution is seen in other species

2. **Structural analysis**: Maps variant to 3D protein structure (if available) and checks:
   - Is it in an active site or binding pocket?
   - Does the substitution change the local structure?
   - Does it affect protein stability?

3. **Physicochemical properties**: Considers:
   - Size of amino acids (small vs. large)
   - Charge (positive, negative, neutral)
   - Hydrophobicity (water-loving vs. water-fearing)

4. **Machine learning prediction**: Combines all features using a naïve Bayes classifier trained on known disease-causing and neutral variants

**Score interpretation**:
- **PolyPhen-2 score**: 0 to 1
- Score > 0.85: "Probably damaging"
- Score 0.45-0.85: "Possibly damaging"
- Score < 0.45: "Benign"

**Example**: The same BRCA1 variant (Glycine → Arginine at position 456)

PolyPhen-2 score = 1.0 ("Probably damaging")

Why?
- Glycine is highly conserved (conservation signal)
- Position 456 is in a DNA-binding domain (structural context)
- Glycine is tiny and flexible; Arginine is large and charged (physicochemical mismatch)

**Strengths**:
- Integrates multiple information sources
- Considers 3D structure when available
- Generally more accurate than SIFT alone
- Provides confidence levels (probably vs. possibly damaging)

**Limitations**:
- More complex (harder to interpret why it made a prediction)
- Requires protein structure (doesn't work well for disordered regions)
- Still imperfect—many false positives and false negatives

### Combining SIFT and PolyPhen-2

In practice, researchers often use both tools and look for agreement:

| SIFT | PolyPhen-2 | Interpretation |
|------|-----------|----------------|
| Deleterious | Probably damaging | High confidence: likely disruptive |
| Tolerated | Benign | High confidence: likely neutral |
| Deleterious | Benign | **Conflicting**: Needs further investigation |
| Tolerated | Probably damaging | **Conflicting**: Needs further investigation |

When tools disagree, it's often because:
- SIFT relies more on conservation; PolyPhen-2 considers structure
- The variant might be in a disordered region (no structure)
- The variant might be human-specific (no orthologs for SIFT)

Conflicting predictions don't mean the tools are broken—they mean the variant is in a gray zone where evolutionary and structural evidence don't align neatly.

### Real Example: Interpreting a Patient Variant

Let's walk through a real diagnostic scenario:

**Patient**: 8-year-old with developmental delays and heart defects

**Variant found**: *TBX5* gene, c.734G>A, p.Gly245Glu
- Changes amino acid at position 245 from Glycine (G) to Glutamate (E)

**Conservation scores**:
- PhyloP: 7.1 (very highly conserved)
- GERP++: 5.9 (strong conservation)

**Protein prediction**:
- SIFT: 0.00 (deleterious)
- PolyPhen-2: 0.99 (probably damaging)

**Interpretation**:
1. Position is highly conserved at DNA level (PhyloP, GERP)
2. Amino acid change is predicted damaging by both protein-level tools
3. Gene *TBX5* is associated with Holt-Oram syndrome (heart and limb defects)
4. Patient's phenotype matches the gene's known function

**Conclusion**: This variant is very likely the cause of the patient's condition. It should be classified as "likely pathogenic" and reported to the clinician.

This is exactly the type of prioritization that makes clinical genomics possible—conservation-based tools help us go from thousands of candidates to a small number of high-confidence answers.

---

## 6.4 The Variant Prioritization Pipeline

Now let's put everything together: how do conservation-based tools fit into a real diagnostic workflow?

### The Challenge Restated

Starting point: **Whole-genome or whole-exome sequencing** from a patient with an undiagnosed condition

Raw data: **4-5 million variants** per genome

Goal: Identify the 1-5 variants causing the patient's disorder

Problem: Can't test all variants experimentally

Solution: **Multi-step filtering pipeline** using conservation and other criteria

### Step-by-Step Pipeline

Here's a typical variant prioritization workflow used in clinical genetics labs:

#### Step 1: Quality Filtering
**Remove low-quality variants**
- Filter out sequencing errors
- Keep only high-confidence variant calls
- Result: ~4 million → ~3.5 million variants

#### Step 2: Frequency Filtering  
**Remove common variants**
- Common variants (MAF > 1% in population databases) are unlikely to cause rare disorders
- Check frequency in gnomAD, 1000 Genomes, etc.
- Result: ~3.5 million → ~50,000 rare variants

#### Step 3: Functional Region Filtering
**Focus on potentially functional variants**
- Exonic variants (change protein sequence)
- Splice site variants (affect RNA splicing)
- Conserved regulatory regions (PhastCons > 0.8)
- Result: ~50,000 → ~5,000 candidates

#### Step 4: Conservation Filtering
**Apply conservation scores**
- Keep variants with PhyloP > 2 OR GERP > 4
- For missense: Keep SIFT < 0.05 OR PolyPhen-2 > 0.45
- Result: ~5,000 → ~500 candidates

#### Step 5: Gene-Phenotype Matching
**Prioritize genes relevant to patient's phenotype**
- Compare patient's symptoms to known gene-disease associations
- Use databases like OMIM, ClinVar, GenCC
- Result: ~500 → ~20-50 high-priority candidates

#### Step 6: Inheritance Pattern Matching
**Consider family structure**
- If patient has unaffected parents: Look for de novo variants
- If patient has affected siblings: Look for recessive variants
- Result: ~20-50 → ~5-15 top candidates

#### Step 7: Literature and Database Review
**Check each candidate**
- Has this specific variant been reported before?
- Are similar variants reported in ClinVar?
- Do case reports describe similar phenotypes?
- Result: ~5-15 → 1-3 diagnostic candidates

### Real Numbers from Clinical Labs

Studies of diagnostic yield in clinical exome sequencing show:

**Starting variants**: 20,000-30,000 (in exonic regions)  
**After frequency filtering**: 500-1,000  
**After conservation filtering**: 100-200  
**After phenotype matching**: 10-30  
**After manual review**: 2-5 reported to clinician  

**Diagnostic rate**: ~25-50% of cases receive a molecular diagnosis

This means even with perfect filtering, about half of patients don't get answers—either because:
1. The causal variant is in a region we don't sequence (deep intronic, regulatory)
2. The disorder is caused by something other than SNVs (e.g., structural variants, repeat expansions)
3. The causal variant is in a gene we don't yet know causes human disorders
4. The disorder isn't primarily genetic

---

## 6.5 Limitations and Caveats

Conservation-based tools are powerful, but they're not perfect. It's critical to understand when they work well and when they fail.

### When Conservation Works Well

**Highly conserved proteins across all life**
- Example: Histones, ribosomal proteins, core metabolic enzymes
- These proteins haven't changed much in a billion years
- Conservation signals are strong and reliable

**Coding variants in well-studied genes**
- Example: *BRCA1*, *CFTR*, *TP53*
- Lots of orthologs available for alignment
- Crystal structures available for PolyPhen-2
- Extensive validation data

**Ancient, essential functions**
- DNA repair, transcription, translation
- Developmental pathways (Hox genes, Notch signaling)
- Selection has been acting for hundreds of millions of years

### When Conservation Fails or Misleads

#### 1. Rapidly Evolving Genes

Some genes evolve quickly by necessity:

**Immune system genes**: MHC genes, immunoglobulins, T-cell receptors
- These NEED to vary to recognize diverse pathogens
- High variation doesn't mean lack of function—it IS the function
- Conservation scores will misleadingly suggest these are non-functional

**Reproductive genes**: Sperm surface proteins, egg coat proteins
- Evolve rapidly due to sexual selection and species barriers
- Variants that look "damaging" by conservation might be normal population variation

**Example**: *PRDM9* (recombination hotspot determination)
- Evolves extremely fast
- Low conservation scores
- But variants in *PRDM9* can cause infertility
- Conservation-based tools underestimate importance

#### 2. Human-Specific or Primate-Specific Genes

**Young genes**: Genes that arose recently in evolution have few or no orthologs in distant species

**Example**: *ARHGAP11B*
- Human-specific gene (not even in chimpanzees)
- Important for human brain development
- SIFT cannot assess variants (no orthologs to align)
- Conservation scores are uninformative

#### 3. Regulatory Variants

**Enhancers and promoters**:
- Often less conserved than coding sequences
- Can have functional variation even at conserved sites
- Conservation scores capture only part of the regulatory logic

**Example**: Lactase persistence enhancer
- The variant that allows adults to digest milk (*LCT* enhancer)
- Arose ~10,000 years ago in some human populations
- Under strong positive selection in dairy-farming populations
- But conservation scores across species show LOW conservation
- Why? Because it's a recent, population-specific adaptation

#### 4. Compensatory Evolution

Sometimes a variant at one position is tolerated ONLY if there's a specific change at another position.

**Example**: tRNA structure
- tRNAs have base-pairing stems
- A mutation disrupting one side of a stem looks damaging
- But if both sides mutate together (compensatory changes), structure is maintained
- Single-position conservation scores miss these co-evolutionary patterns

### False Positives and False Negatives

**False positives** (predicted damaging but actually neutral):
- ~20-40% of variants predicted "damaging" by SIFT/PolyPhen-2 are actually benign when tested experimentally
- Why? Tools are conservative—they err on the side of caution
- Clinical labs must validate before reporting

**False negatives** (predicted benign but actually damaging):
- Rarer (~10-20%) but still happen
- Often due to variants in poorly conserved genes or regulatory regions
- Can lead to missed diagnoses

**Practical implication**: Never rely solely on conservation scores. Always integrate with:
- Clinical phenotype
- Family history / inheritance pattern
- Functional studies when possible
- Literature evidence

---

## 6.6 Beyond Single-Variant Predictions: Constraint Metrics

Conservation scores assess individual positions, but what if we zoom out and ask: **Which genes tolerate variation, and which don't?**

This question has led to **gene-level constraint metrics**—scores that summarize how much variation a gene tolerates overall.

### The Concept of Constraint

**Constrained genes**: Accumulate very few loss-of-function variants in the population
- Why? Because such variants reduce fitness (survival or reproduction)
- Natural selection removes them
- These genes are under strong selective **constraint**

**Unconstrained genes**: Accumulate loss-of-function variants freely
- Why? Because losing one copy (or even both) doesn't harm fitness much
- These genes are NOT under strong selective constraint

The key insight: **Genes under strong constraint are more likely to cause disorders when mutated**.

### pLI (Probability of Loss-of-Function Intolerance)

**Developed by**: Exome Aggregation Consortium (ExAC) → Genome Aggregation Database (gnomAD)

**What it measures**: The probability that a gene is intolerant to loss-of-function (LoF) variants

**How it works**:
1. Count expected number of LoF variants in a gene (based on gene size and mutation rate)
2. Count observed number of LoF variants in the population (from gnomAD: 125,000+ exomes)
3. Compare expected vs. observed

**If observed << expected**: Gene is intolerant → High constraint → pLI close to 1  
**If observed ≈ expected**: Gene is tolerant → Low constraint → pLI close to 0

**Interpretation**:
- pLI > 0.9: Gene is highly intolerant to LoF (haploinsufficient)
- pLI < 0.1: Gene is tolerant to LoF

**Example**:
- *MECP2* (Rett syndrome gene): pLI = 1.0
  - Expected LoF variants: 15
  - Observed LoF variants: 0
  - Interpretation: Every LoF variant is being removed by selection → gene is essential

- *OR4F5* (olfactory receptor): pLI = 0.0
  - Expected LoF variants: 2
  - Observed LoF variants: 12
  - Interpretation: LoF variants are tolerated → gene is not essential

**Clinical application**: If a patient has a de novo LoF variant in a gene with pLI = 1.0, that's strong evidence the variant is causative.

### LOEUF (Loss-of-Function Observed/Expected Upper Bound Fraction)

**What it measures**: Similar to pLI but provides a continuous score rather than a probability

**How it works**:
- Calculates ratio: observed LoF / expected LoF
- Adds confidence intervals to account for statistical uncertainty
- LOEUF = upper bound of the confidence interval

**Interpretation**:
- LOEUF < 0.35: Strong constraint (top ~20% most constrained genes)
- LOEUF > 1.0: Little to no constraint

**Advantage over pLI**: LOEUF provides a continuous spectrum and is more robust for genes with small numbers of variants

**Example**:
- *BRCA1*: LOEUF = 0.11 (very constrained)
- *CFTR*: LOEUF = 0.14 (very constrained)
- *APOE*: LOEUF = 0.88 (less constrained—some LoF is tolerated)

---

## 6.7 Putting It All Together: A Real Diagnostic Case

Let's walk through a complete example to see how conservation-based tools work in practice.

### Patient Presentation

**Patient**: 3-year-old girl  
**Symptoms**:
- Global developmental delay (can't walk or speak)
- Seizures starting at 6 months
- Microcephaly (small head)
- Brain MRI shows reduced white matter

**Family history**: Parents unaffected, no family history of neurological disorders

**Initial diagnosis**: Unknown

### Sequencing and Initial Filtering

**Test ordered**: Trio exome sequencing (patient + both parents)

**Initial variants called**: 22,458 exonic variants

**Step 1 - Frequency filtering** (MAF < 0.1%): Remaining: 1,247 rare variants

**Step 2 - Inheritance filtering** (de novo or recessive):
- De novo variants: 3 variants
- Homozygous variants: 12 variants
- Compound heterozygous: 4 gene pairs
- **Total candidates: 23 variants across 16 genes**

### Applying Conservation-Based Tools

#### Candidate 1: *KCNQ2* c.638G>A (p.Arg213Gln)

**Type**: Missense, de novo

**Conservation**:
- PhyloP: 6.8 (very high)
- GERP++: 5.9 (high)

**Protein predictions**:
- SIFT: 0.001 (deleterious)
- PolyPhen-2: 0.98 (probably damaging)

**Gene constraint**:
- pLI: 1.0 (completely intolerant to LoF)
- LOEUF: 0.08 (top 5% most constrained)

**Gene function**: Potassium channel, critical for neuronal excitability

**Known gene-disease association**: *KCNQ2* mutations cause epileptic encephalopathy (matches patient phenotype!)

**ClinVar**: Similar variants reported as "pathogenic"

**Assessment**: **STRONG CANDIDATE** - This looks like the causative variant

#### Candidate 2: *TTN* c.53479G>T (p.Val17827Phe)

**Type**: Missense, de novo

**Conservation**:
- PhyloP: 1.2 (moderate)
- GERP++: 2.1 (weak)

**Protein predictions**:
- SIFT: 0.12 (tolerated)
- PolyPhen-2: 0.35 (benign)

**Gene function**: Giant muscle protein

**Assessment**: **LOW PRIORITY** - Likely benign variant

#### Candidate 3: *SCN2A* c.2447A>G (p.Glu816Gly)

**Type**: Missense, de novo

**Conservation**:
- PhyloP: 5.2 (high)
- GERP++: 5.1 (high)

**Protein predictions**:
- SIFT: 0.02 (deleterious)
- PolyPhen-2: 0.89 (probably damaging)

**Gene function**: Sodium channel, critical for action potential generation

**Known gene-disease association**: *SCN2A* mutations cause epilepsy and developmental delay (excellent match!)

**Assessment**: **STRONG CANDIDATE** - Also looks causative

### Resolution

Two strong candidates: *KCNQ2* and *SCN2A*. Both fit the phenotype. Both have high conservation. Both are predicted damaging.

**Next steps**:

1. **Functional studies**: Test both variants in cell culture
   - *KCNQ2* variant: Patch-clamp shows loss of channel function → **Confirmed pathogenic**
   - *SCN2A* variant: Patch-clamp shows normal function → **Likely benign** (surprising!)

**Final Diagnosis**: *KCNQ2*-related epileptic encephalopathy

**Lessons from This Case**:
1. Conservation helped narrow candidates: From 22,000 variants to 2 strong candidates
2. Conservation isn't perfect: *SCN2A* variant looked damaging by all predictions but turned out to be neutral
3. Phenotype matching is crucial
4. Functional validation matters when possible
5. Clinical context wins: the combination provides confident diagnostic conclusions

---

## Summary

### Key Takeaways

**The power of evolutionary information:**
- Evolution has tested billions of mutations over millions of years—we can use this information without doing experiments ourselves
- High conservation = functional importance; positions that stay unchanged across species are probably doing something critical
- Conservation works at multiple levels: nucleotides (PhyloP, GERP), amino acids (SIFT, PolyPhen-2), and genes (pLI, LOEUF)

**DNA-level conservation scores:**
- PhyloP measures single-nucleotide conservation; positive scores indicate slower evolution than expected (conserved positions)
- GERP++ counts "rejected substitutions"—how many mutations natural selection has removed at each position
- PhastCons identifies conserved elements (stretches of consecutive conserved positions), useful for finding regulatory regions
- Typical thresholds: PhyloP > 2, GERP > 4, PhastCons > 0.8 for strong conservation

**Protein-level predictions:**
- SIFT asks "Is this substitution seen in other species?" If no → probably damaging (score < 0.05 = deleterious)
- PolyPhen-2 integrates conservation + structure + physicochemical properties (score > 0.85 = probably damaging)
- Tools can disagree (~20% of variants); use both scores plus clinical judgment

**Gene-level constraint:**
- pLI (probability of loss-of-function intolerance): pLI > 0.9 means gene is haploinsufficient
- LOEUF (loss-of-function observed/expected upper bound): LOEUF < 0.35 indicates strong constraint
- Genes under strong constraint are more likely to cause disorders when mutated

**Limitations to remember:**
- Conservation-based tools have ~20-30% false positive rate
- Tools fail for: rapidly evolving genes (immune, reproductive), human-specific genes, recent adaptations
- Always integrate with: population frequency, clinical phenotype, family history, functional studies

---

<details>
<summary><strong>📖 Key Terms</strong></summary>

| Term | Definition |
|------|-----------|
| **Conservation score** | A numerical measure of how much a DNA position or amino acid has been preserved across species through evolution |
| **PhyloP** | Conservation score measuring whether a position evolves slower (positive score) or faster (negative score) than expected by chance |
| **GERP++ (Genomic Evolutionary Rate Profiling)** | Score representing the number of mutations rejected by natural selection at a given position |
| **PhastCons** | Score (0-1) representing the probability that a position belongs to a conserved element |
| **SIFT (Sorting Intolerant From Tolerant)** | Tool predicting whether an amino acid substitution is tolerated (score ≥ 0.05) or deleterious (score < 0.05) based on evolutionary conservation |
| **PolyPhen-2 (Polymorphism Phenotyping v2)** | Machine learning tool integrating conservation, structure, and physicochemical properties to predict variant impact |
| **Missense variant** | A genetic variant that changes one amino acid to another in a protein sequence |
| **Multiple sequence alignment (MSA)** | Arrangement of DNA or protein sequences from multiple species to identify conserved and variable positions |
| **Ortholog** | The same gene in different species (e.g., human insulin vs. mouse insulin) |
| **pLI (Probability of Loss-of-Function Intolerance)** | Score (0-1) indicating how intolerant a gene is to loss-of-function variants; scores > 0.9 suggest haploinsufficiency |
| **LOEUF (Loss-of-Function Observed/Expected Upper Bound)** | Ratio of observed to expected loss-of-function variants in a gene; scores < 0.35 indicate strong constraint |
| **Constraint** | The degree to which a gene or genomic region is intolerant to variation due to selection against deleterious mutations |
| **Haploinsufficiency** | Condition where one functional copy of a gene is not enough for normal function |

</details>

---

## Test Your Understanding

1. Explain why a highly conserved DNA position is more likely to be functionally important than a non-conserved position. What evolutionary forces create this pattern?

2. Compare and contrast PhyloP, GERP++, and PhastCons. When might each score be most useful?

3. A variant has SIFT score of 0.01 (deleterious) but PolyPhen-2 score of 0.2 (benign). What might explain this discrepancy? How would you investigate further?

4. Why do immune system genes and reproductive genes often have low conservation scores? Does this mean variants in these genes are harmless?

5. Explain the difference between pLI and LOEUF. Why might a gene have pLI = 0.98 but still occasionally have loss-of-function variants in the population?

6. A patient has a de novo missense variant in a gene with pLI = 0.02 and LOEUF = 1.8. How would you interpret this finding? Would you prioritize this as a causative variant?

7. Why might conservation-based tools fail to identify pathogenic variants in human-specific genes or recently evolved genes?

8. Describe a scenario where a variant at a non-conserved position could still be pathogenic. How would you identify such variants?
