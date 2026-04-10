# Chapter 7: Machine Learning Ensemble Methods for Variant Interpretation

**[Interactive: Chapter 7](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter7.html)**

## Opening Vignette

Dr. Chen stares at her spreadsheet with growing frustration. She's analyzing a variant in the *SCN1A* gene from a patient with developmental delays and seizures. The problem? Every prediction tool is giving her different answers.

SIFT says the variant is "tolerated" (score: 0.12). PolyPhen-2 calls it "probably damaging" (score: 0.89). PhyloP shows moderate conservation (score: 2.1). The variant changes a leucine to proline in a transmembrane domain—clearly important—but it appears once in gnomAD with 125,000 alleles sequenced. Is that "rare enough" to be disorder-causing?

She has 63 different pieces of information about this single variant: conservation scores from multiple methods, functional predictions, allele frequencies in different populations, proximity to splice sites, predicted protein stability changes, and more. Each tool captures something real about the variant, but they often disagree. Some variants score high on conservation but low on predicted functional impact. Others show the opposite pattern.

What Dr. Chen needs isn't another single prediction tool—she needs a way to combine all these signals intelligently. She needs to know: when SIFT and PolyPhen-2 disagree, which should she trust? When they agree, how much more confident should she be? And can she somehow use all 63 pieces of information together, letting each contribute what it does best?

This is exactly what machine learning ensemble methods were designed to solve.

---

## The Biological Challenge

Genetic variant interpretation faces a fundamental problem: no single feature perfectly predicts functional impact. Conservation matters, but not all conserved positions are functionally critical. Predicted protein changes matter, but not all amino acid substitutions affect function. Allele frequency matters, but rare variants aren't always disorder-causing.

In Chapter 6, we explored individual prediction tools—each capturing one aspect of variant biology. But variants are complex, multifaceted entities. A variant in a highly conserved residue might still be neutral if it preserves chemical properties. A variant at a poorly conserved position might still have functional impact if it disrupts a critical binding site.

**Why we can't just pick the "best" tool:**

First, there is no single "best" tool. Different tools excel at different tasks:
- Conservation scores work well for ancient, slowly-evolving proteins
- Functional predictors work well for enzymes and structured domains
- Population frequencies work well for common variants but are unreliable for rare ones
- Splicing predictors work well near exon-intron boundaries but nowhere else

Second, the scale of the problem is enormous. Whole-genome sequencing identifies 4-5 million variants per person. Whole-exome sequencing still yields 20,000-30,000 variants. Even after filtering to coding variants in known disorder genes, hundreds of candidates remain.

Third, individual tools make errors in predictable patterns. SIFT tends to be conservative (fewer false positives but more false negatives). PolyPhen-2 tends to be aggressive (fewer false negatives but more false positives). Understanding and compensating for these systematic biases requires analyzing thousands of validated examples—a job for machine learning.

**The computational solution:** Instead of choosing one tool or manually weighing evidence, we can train machine learning models to automatically:
- Learn which features are most informative
- Learn when each tool can be trusted
- Combine multiple weak signals into strong predictions
- Adapt to different variant types (missense, splice, regulatory)
- Continuously improve as more validated examples become available

This approach, called **ensemble learning**, treats each prediction tool as a "voter" in a committee. Think of it like a scientific panel review—no single reviewer decides; the committee vote reduces individual bias. The machine learning model learns how much weight to give each voter, under what circumstances, and how to combine their votes into a final prediction. The result: predictions that are more accurate than any single tool alone.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- [ ] Explain the concept of ensemble learning and why it improves variant prediction
- [ ] Describe how CADD integrates 63 annotations into a single deleteriousness score
- [ ] Understand the difference between linear models (CADD) and neural networks (DANN)
- [ ] Compare meta-predictors (MetaLR, MetaSVM) that combine existing tools
- [ ] Evaluate the specialized design of REVEL for rare missense variants
- [ ] Interpret ensemble scores in clinical contexts and understand their limitations
- [ ] Apply ensemble methods to prioritize variants in whole-exome sequencing data

---

## 7.1 The Wisdom of Crowds: Why Ensemble Learning Works

Imagine you're trying to estimate the weight of a large pumpkin at a county fair. You could:
1. Ask one expert judge
2. Ask multiple people and take the average
3. Ask multiple people, but weight experts' guesses more heavily

Option 3 is ensemble learning. Even if no single person knows the exact weight, combining many estimates—especially when you know who tends to overestimate or underestimate—often gives remarkably accurate results.

The same principle applies to variant prediction. We have dozens of "expert judges" (prediction tools), each with its own strengths and biases. Ensemble methods learn to combine them optimally.

### 7.1.1 The Power of Combination

Consider a simple example. Tool A correctly identifies 70% of variants with functional impact but has a 20% false positive rate. Tool B correctly identifies 65% but has only a 10% false positive rate. If you simply pick the "better" tool (A), you miss opportunities:

- When A and B both predict functional impact → very likely true
- When A predicts impact but B doesn't → less certain
- When neither predicts impact → very likely neutral
- When B predicts impact but A doesn't → might be worth investigating

An ensemble method can learn these patterns from thousands of validated examples, automatically discovering the best way to combine tools.

### 7.1.2 Three Types of Ensemble Methods

**1. Feature Integration (CADD)**  
Combines raw features (conservation scores, frequencies, structural predictions) directly. The model learns which features matter and how to weight them. Think of this as giving the model the raw ingredients and letting it learn the recipe.

**2. Meta-Prediction (MetaLR, MetaSVM, REVEL)**  
Combines outputs from existing prediction tools. Instead of learning from raw features, these models learn from what other tools predict. Think of this as combining expert opinions.

**3. Neural Network Ensembles (DANN)**  
Uses neural networks instead of linear models to capture complex, nonlinear relationships between features. Can learn interactions between features that linear models miss.

Each approach has trade-offs in interpretability, computational cost, and accuracy. We'll explore each in detail.

---

## 7.2 CADD: Combined Annotation Dependent Depletion

CADD (pronounced "cad") was one of the first successful ensemble methods for variant interpretation. Published in 2014 and continuously updated, CADD scores are now among the most widely used in clinical genetics.

### 7.2.1 The Core Idea

CADD asks a clever question: **How different is this variant from what we'd expect to see in typical human genetic variation?**

The reasoning: variants that cause disorders are rare in the population precisely because natural selection removes them. If we can quantify how "unusual" a variant is compared to common variation, we can estimate its likelihood of having functional impact.

CADD doesn't directly predict "disorder-causing" vs "neutral." Instead, it predicts "looks like common variants" vs "looks like rare, selected-against variants." This subtle difference makes CADD applicable across many contexts.

### 7.2.2 Training Strategy

CADD uses a brilliant training approach:

**Positive examples (variants likely to be selected against):**
- Simulated de novo mutations (randomly generated variants that haven't been through natural selection yet)
- These represent what the genome would look like without selective pressure

**Negative examples (variants tolerated by selection):**
- Common variants from population databases (allele frequency > 1%)
- These passed through generations of selection, so they're probably neutral

The model learns to distinguish "looks like it hasn't been filtered by selection" (potentially functional) from "looks like it survived selection" (probably neutral).

### 7.2.3 The 63 Annotations

CADD integrates 63 different features across multiple categories:

**Sequence Conservation (15 annotations)**
- PhyloP scores at different evolutionary depths (vertebrate, mammal, primate)
- PhastCons conservation
- GERP scores
- These capture how much the position is conserved across species

**Functional Predictions (12 annotations)**
- SIFT scores
- PolyPhen-2 scores
- Predicted protein structure changes
- Changes in predicted protein stability
- These estimate direct functional effects

**Regulatory Annotations (20 annotations)**
- Overlap with DNase hypersensitive sites
- Overlap with transcription factor binding sites
- Histone modification patterns (H3K4me1, H3K4me3, H3K27ac, etc.)
- CpG islands
- These identify variants in regulatory regions

**Transcript Annotations (8 annotations)**
- Gene model information
- Exon/intron boundaries
- Distance to splice sites
- Predicted splicing changes
- These provide genomic context

**Population Genetics (8 annotations)**
- Local nucleotide diversity
- Background selection statistics
- Recombination rates
- These capture regional genetic variation patterns

CADD doesn't just use these 63 features—it learns their relative importance from data.

### 7.2.4 The Model

CADD uses a linear classifier that learns weights for each feature and combines them into a final score. You can think of this as a weighted vote: each of the 63 features casts a "vote," and CADD learns how much to trust each voter.

> **[선택: 수식으로 보면]**
>
> CADD uses a **Support Vector Machine (SVM)** — a type of linear classifier. An SVM learns weights (*w*) for each feature (*x*) and combines them:
>
> score = *w*₁*x*₁ + *w*₂*x*₂ + ... + *w*₆₃*x*₆₃ + *b*
>
> Where *x*ᵢ are the 63 feature values, *w*ᵢ are learned weights (positive weights increase the score; negative weights decrease it), and *b* is a bias term. For example, if PhyloP conservation is *x*₁ = 5.2 and its learned weight is *w*₁ = 0.3, it contributes 0.3 × 5.2 = 1.56 to the final score.

### 7.2.5 CADD Scores: Interpretation

CADD outputs scores on a **phred-like scale**:
- CADD = 10 → variant is in the top 10% of deleteriousness (one in 10)
- CADD = 20 → top 1% (one in 100)
- CADD = 30 → top 0.1% (one in 1,000)
- CADD = 40 → top 0.01% (one in 10,000)

Higher scores indicate greater predicted deleteriousness.

**Clinical interpretation guidelines:**
- CADD < 15: likely neutral
- CADD 15-20: uncertain significance
- CADD 20-30: likely functional impact
- CADD > 30: very likely functional impact

**Important caveats:**
- CADD scores are continuous—there's no hard cutoff
- Scores should be interpreted in clinical context
- High CADD scores don't prove causation
- Low CADD scores don't rule out functional impact in all cases

### 7.2.6 Example: CADD in Practice

Think of a CADD score like a pathologist grading a biopsy — it combines multiple features (conservation, structural impact, population frequency) into a single pathogenicity score. A high CADD score doesn't make the final diagnosis, but it tells you which specimen to look at more closely.

Let's return to Dr. Chen's *SCN1A* variant:

```
Variant: chr2:166,848,542 A>C (GRCh38)
Gene: SCN1A
Change: Leu1231Pro (leucine to proline at position 1,231)
CADD score: 28.3
```

What does CADD = 28.3 tell us?

1. **Rank:** This variant is in the top 0.15% of all possible variants by deleteriousness
2. **Support from features:**
   - High PhyloP conservation (contributes positively)
   - PolyPhen-2 "probably damaging" (contributes positively)
   - Changes leucine to proline in transmembrane domain (structural disruption, contributes positively)
   - Low population frequency (contributes positively)
3. **Interpretation:** The score of 28.3 suggests this variant likely has functional impact and warrants further investigation

However, CADD alone doesn't establish causation. We'd still need segregation with the disorder in the family, functional studies, absence in large control databases, and other clinical evidence.

---

## 7.3 DANN: Deep Learning Improves on CADD

CADD uses a linear model, which assumes features combine additively. But biological relationships are often nonlinear and interactive. For example:
- A variant might only matter if it's in a conserved region AND disrupts protein structure
- Conservation might matter more in some functional domains than others
- Frequency thresholds might depend on gene constraint

**DANN (Deep Annotation of geNetic variatioN)** addresses this by using a neural network instead of a linear model.

### 7.3.1 From Linear to Nonlinear

CADD computes: score = *w*₁*x*₁ + *w*₂*x*₂ + ... + *b*

DANN computes: score = neural_network(*x*₁, *x*₂, ..., *x*₆₃)

The neural network can learn complex, nonlinear relationships between the 63 features—interactions that a linear model cannot capture.

**What does this enable?**

1. **Feature interactions:** DANN can learn "if conservation is high AND the amino acid change is drastic, then increase the score dramatically"
2. **Context-dependent weighting:** DANN can learn "conservation matters more for transmembrane domains than for flexible loops"
3. **Nonlinear thresholds:** DANN can learn more nuanced boundaries between damaging and neutral

### 7.3.2 The DANN Architecture

DANN uses a neural network with input, hidden, and output layers. The same 63 features used by CADD serve as inputs. The network was trained with the same strategy as CADD (simulated vs. common variants).

### 7.3.3 DANN vs CADD Performance

DANN shows modest but consistent improvements over CADD:

**On ClinVar variants (known functional impact):**
- CADD: 77% correctly ranked above median
- DANN: 79% correctly ranked above median

**On validated neutral variants:**
- CADD: 74% correctly ranked below median
- DANN: 76% correctly ranked below median

The improvement seems small (2-3%), but at the scale of clinical genetics—where thousands of variants must be prioritized—this translates to hundreds of additional correctly classified variants.

### 7.3.4 The Interpretability Trade-off

CADD's linear model is interpretable: you can see exactly how much each feature contributed to the final score. DANN's neural network is a "black box": you can't easily explain why it gave a specific score.

This matters in clinical genetics. When counseling patients, being able to say "this variant scores high because it's in a highly conserved region, disrupts protein structure, and is absent from population databases" is more compelling than "the neural network gave it a high score."

**Current practice:**
- Use CADD for primary scoring and reporting (interpretability)
- Use DANN as a secondary check for difficult cases
- Report both scores when they disagree substantially

---

## 7.4 Meta-Predictors: Combining Existing Tools

CADD and DANN work directly with 63 raw features. But what if you want to combine existing prediction tools (SIFT, PolyPhen-2, MutationTaster, etc.) without retraining everything from scratch?

**Meta-predictors** take the outputs of multiple existing tools as inputs and learn to combine them optimally.

### 7.4.1 The Meta-Prediction Strategy

Think of meta-prediction as a "second opinion" approach:

1. Run the variant through many existing tools:
   - SIFT → score 1
   - PolyPhen-2 → score 2
   - PROVEAN → score 3
   - MutationTaster → score 4
   - ... (and so on)

2. Train a machine learning model on these tool scores:
   - Inputs: scores from existing tools
   - Labels: known functional impact / neutral variants from ClinVar
   - Learn: which tools to trust, when to trust them, how to combine them

3. Output a meta-score that's more accurate than any individual tool

### 7.4.2 MetaLR and MetaSVM

**MetaLR** and **MetaSVM** are sister methods that differ only in their machine learning algorithm:
- **MetaLR:** Uses logistic regression (a simpler, more interpretable model)
- **MetaSVM:** Uses support vector machines (can capture slightly more complex patterns)

Both integrate scores from **10 existing tools:**
1. SIFT
2. PolyPhen-2 (HumDiv)
3. PolyPhen-2 (HumVar)
4. LRT (Likelihood Ratio Test)
5. MutationTaster
6. MutationAssessor
7. FATHMM
8. PROVEAN
9. VEST
10. CADD

Notice that CADD itself is one of the inputs—meta-predictors can use other ensemble methods as features!

### 7.4.3 Interpreting Meta-Scores

MetaLR and MetaSVM output scores from 0 to 1:
- **0.0-0.3:** Likely neutral
- **0.3-0.5:** Uncertain
- **0.5-0.7:** Likely functional impact
- **0.7-1.0:** Very likely functional impact

**Clinical usage:**
- MetaSVM score > 0.83 → strong evidence for functional impact
- MetaLR score > 0.81 → strong evidence for functional impact

These thresholds are chosen to maximize specificity (minimize false positives) in clinical settings.

### 7.4.4 Advantages and Limitations

**Advantages:**
- Easy to implement (just run existing tools and combine scores)
- Leverages decades of tool development
- Each tool provides different biological perspective
- Relatively interpretable (can see which tools drove the prediction)

**Limitations:**
- Dependent on quality of input tools
- If all tools have the same blind spot, the meta-predictor inherits it
- Cannot learn from raw features (limited by what tools provide)
- Computationally expensive (must run 10 different tools)

---

## 7.5 REVEL: Specialized for Rare Missense Variants

Most ensemble methods treat all variants equally. But clinical genetics has a specific pain point: **rare missense variants** in exome sequencing data.

Why are rare missense variants challenging?
1. Too rare to have reliable population frequency data
2. Too new to have functional studies or clinical reports
3. Missense changes are subtle (just one amino acid)—harder to predict than truncating variants
4. Occur in variable domains where conservation is weak

**REVEL (Rare Exome Variant Ensemble Learner)** was specifically designed for this problem.

### 7.5.1 REVEL's Design Philosophy

REVEL makes several key choices:

**1. Focus exclusively on missense variants**  
Doesn't try to handle splice variants, indels, or regulatory variants. Allows optimization for the specific challenges of missense prediction.

**2. Use only rare variants in training**  
- Positive examples: ClinVar variants associated with Mendelian disorders (allele frequency < 0.1%)
- Negative examples: rare variants from ESP (allele frequency < 0.1%)
- This better reflects the clinical use case: interpreting rare variants in patients

**3. Integrate a specific set of 13 tools**  
Selected for complementary strengths in missense variant prediction: MutPred, FATHMM v2.3, VEST 3.0, PolyPhen-2, SIFT, PROVEAN, MutationAssessor, MutationTaster, LRT, GERP++, SiPhy, phyloP, and phastCons.

**4. Use a random forest classifier**  
Random forests are ensemble methods within an ensemble method!

### 7.5.2 Random Forests: Ensemble of Ensembles

**What is a random forest?**

Imagine you want to classify variants, so you train a decision tree:

```
                 Is SIFT < 0.05?
                /              \
              Yes               No
              /                  \
    Is PolyPhen > 0.9?     Is PhyloP > 3?
       /      \              /        \
     Yes      No           Yes        No
     /         \            /          \
  Functional  Neutral  Functional   Neutral
```

One tree can overfit to training data and make errors. A **random forest** builds hundreds of trees, each trained on:
- A random subset of variants (sampling with replacement)
- A random subset of features

Then votes: majority vote across all trees gives the final prediction.

**Why this works:**
- Different trees make different errors
- Averaging over many trees reduces overfitting
- Captures complex, nonlinear patterns
- Robust to noisy features

REVEL trains 500 decision trees and averages their predictions.

### 7.5.3 REVEL Scores: Interpretation

REVEL outputs scores from 0 to 1, calibrated specifically for rare missense variants:
- **REVEL > 0.75:** Strong evidence for functional impact
- **REVEL 0.5-0.75:** Moderate evidence
- **REVEL 0.25-0.5:** Uncertain
- **REVEL < 0.25:** Likely neutral

**Performance on rare missense variants:**
- Sensitivity: 75% (correctly identifies variants with functional impact)
- Specificity: 92% (correctly identifies neutral variants)

This outperforms individual tools and general-purpose ensemble methods on rare missense variants specifically.

### 7.5.4 When to Use REVEL

**Use REVEL for:**
- Rare missense variants (MAF < 0.1%)
- Variants in coding exons
- Patient variant interpretation in exome sequencing
- Prioritizing candidate disorder-causing variants

**Don't use REVEL for:**
- Common variants (MAF > 1%)—not designed for these
- Splice-site variants—use SpliceAI or MaxEntScan instead
- Indels—use CADD or DANN instead
- Regulatory variants—use DeepSEA or Enformer (Chapters 7-8)

**Clinical tip:** Many labs use a tiered approach:
1. CADD for initial filtering (all variant types)
2. REVEL for refining missense variant calls
3. Specialized tools for splice/regulatory variants
4. Manual review of high-scoring variants

---

## 7.6 Comparing Ensemble Methods

Let's compare the ensemble methods we've covered:

| Method | Type | # Features | Variant Types | Best For | Output Scale |
|--------|------|-----------|---------------|----------|--------------|
| **CADD** | Linear (SVM) | 63 | All | General screening | Phred (10-40) |
| **DANN** | Neural network | 63 | All | Complex cases | 0-1 |
| **MetaLR** | Meta (logistic) | 10 tools | All | Combining tools | 0-1 |
| **MetaSVM** | Meta (SVM) | 10 tools | All | Combining tools | 0-1 |
| **REVEL** | Meta (random forest) | 13 tools | Missense only | Rare missense | 0-1 |

### 7.6.1 Performance Comparison

On a benchmark set of 5,000 ClinVar variants and 5,000 common variants:

| Method | Sensitivity | Specificity | AUC |
|--------|------------|-------------|-----|
| SIFT | 66% | 82% | 0.84 |
| PolyPhen-2 | 72% | 79% | 0.87 |
| CADD | 77% | 88% | 0.91 |
| DANN | 79% | 89% | 0.92 |
| MetaSVM | 81% | 90% | 0.93 |
| REVEL* | 75% | 92% | 0.93 |

*REVEL evaluated only on rare missense variants

**Key observations:**
1. Ensemble methods outperform individual tools
2. Neural networks (DANN) slightly outperform linear models (CADD)
3. Meta-predictors (MetaSVM) perform best overall
4. REVEL optimizes for specificity at the cost of sensitivity
5. No method is perfect—all make errors

### 7.6.2 Agreement Between Methods

When methods agree, confidence is high. For a random set of 1,000 rare variants in disorder genes:
- CADD and DANN agree (both high or both low): 87%
- CADD and MetaSVM agree: 82%
- All four agree: 71%

The 29% where they disagree are the challenging cases that require expert review.

---

## 7.7 Practical Considerations: Using Ensemble Methods

### 7.7.1 Choosing the Right Tool

**Decision tree for method selection:**

```
Is the variant missense?
├─ Yes → Is it rare (MAF < 0.1%)?
│        ├─ Yes → Use REVEL (primary) + CADD (secondary)
│        └─ No → Use CADD
└─ No → What type?
         ├─ Truncating (nonsense, frameshift) → likely functional; confirm with gene constraint
         ├─ Splice-region → Use SpliceAI
         └─ Regulatory → Use DeepSEA or Enformer (Chapters 7-8)
```

### 7.7.2 Interpreting Conflicting Scores

When ensemble methods disagree, consider:

**1. Check the raw features**  
If CADD is high but REVEL is low: examine conservation, structure, and frequency independently. Often reveals why tools disagree.

**2. Consider variant type**  
If splice variant, don't trust REVEL (designed for missense). If in regulatory region, even high CADD may miss enhancer disruption.

**3. Examine population data**  
If variant appears multiple times in gnomAD, likely neutral regardless of scores. If completely absent and in constrained gene, scores matter more.

**4. Look for functional studies**  
Published experiments trump predictions. Check OMIM, ClinVar, PubMed for functional data.

**5. When in doubt, use multiple lines of evidence**  
- Segregation in family
- Multiple affected individuals with same variant
- Functional assays (luciferase, electrophysiology, etc.)
- Animal models

---

## Summary

**Key Takeaways:**

- Ensemble learning combines multiple weak signals into strong predictions, outperforming individual tools
- CADD integrates 63 genomic annotations using a linear model, providing broadly applicable deleteriousness scores on a phred-like scale
- DANN uses neural networks to capture nonlinear feature interactions, improving on CADD's linear model by a few percentage points
- Meta-predictors (MetaLR, MetaSVM) combine outputs from existing tools, learning which tools to trust and when
- REVEL specializes in rare missense variants, using random forests (500 decision trees voting together) to achieve high specificity in clinical settings
- The ensemble approach is like a scientific panel review—no single reviewer decides; the committee vote reduces individual bias
- No single method is perfect; combining multiple ensemble methods and incorporating clinical context improves accuracy
- Ensemble scores help prioritize variants but don't replace comprehensive evaluation including segregation, functional studies, and clinical assessment

---

<details>
<summary><strong>📖 Key Terms</strong></summary>

| Term | Definition |
|------|-----------|
| **Deleteriousness score** | A quantitative measure of how likely a variant is to have functional impact or be selected against |
| **Ensemble learning** | Machine learning approach that combines multiple models or features to improve prediction accuracy |
| **Feature integration** | Ensemble strategy that combines raw data (e.g., conservation scores, frequencies) directly |
| **Meta-prediction** | Ensemble strategy that combines outputs from existing prediction tools |
| **Phred-scaled score** | Logarithmic scale where score X means "top 10^(-X/10)" (e.g., 20 = top 1%, 30 = top 0.1%) |
| **Random forest** | Ensemble method that combines predictions from many decision trees to reduce overfitting |
| **Support Vector Machine (SVM)** | Linear classifier that finds the optimal boundary between classes in high-dimensional space |
| **Training data** | Validated examples (with known labels) used to train machine learning models |
| **Sensitivity** | Proportion of variants with functional impact correctly identified (true positive rate) |
| **Specificity** | Proportion of neutral variants correctly identified (true negative rate) |

</details>

---

## Test Your Understanding

1. Why does combining multiple prediction tools often work better than using the single "best" tool? Explain using the concept of complementary information.

2. CADD uses common variants as negative training examples, reasoning that natural selection has already filtered out variants with functional impact. What are potential limitations of this assumption? When might it fail?

3. DANN improves on CADD by using neural networks instead of linear models. Explain in biological terms what kinds of relationships between features a neural network can learn that a linear model cannot.

4. MetaSVM and REVEL both use ensemble learning but perform differently on rare missense variants. Why does REVEL outperform MetaSVM specifically for this variant class?

5. A variant has CADD = 32, DANN = 0.95, MetaSVM = 0.45, and REVEL = 0.22. What might explain these conflicting scores? How would you interpret this variant?

6. Why is interpretability important in clinical variant interpretation? Describe a situation where you'd prefer CADD over DANN despite DANN's higher accuracy.

7. Many ensemble methods are trained on ClinVar variants, which are heavily biased toward coding variants in well-studied genes. How might this affect their performance on regulatory variants or variants in poorly characterized genes?

8. If you were designing a new ensemble method for structural variants (deletions, duplications > 1 kb), what features would you integrate and why?
