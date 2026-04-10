# Chapter 8: The Rise of Deep Learning in Genomics

**[Interactive: Chapter 8](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter8.html)**

For decades, computational biologists wrote rules by hand. They cataloged the DNA sequences where transcription factors were known to bind, assembled them into databases like JASPAR, and built scoring algorithms that measured how closely a new sequence matched these curated patterns. If a sequence looked like a known binding site, it probably was one. The approach had a real logic to it, and it worked — for the patterns that were already known.

But the genome kept producing surprises. Enhancers that drove powerful tissue-specific expression without matching any motif in any database. Regulatory elements that functioned only at a specific developmental stage, or only in one of two otherwise identical cell types. Noncoding variants that disrupted gene regulation through mechanisms that nobody had articulated, let alone cataloged. The rule-based approach was not wrong so much as incomplete: it could recognize what researchers had already described, but it had no way to discover what they had not yet thought to look for.

Somewhere around 2015, a quiet revolution began. A handful of groups — working mostly on chromatin accessibility and transcription factor binding — asked a different question. Instead of telling a computer which patterns to look for, what if you showed it millions of sequences with known regulatory activity and let it find the patterns itself? No curated motif database. No manually encoded rules. Just raw DNA sequence as input, experimental measurements as output, and a deep neural network tasked with finding whatever structure connected the two.

The results were striking. These models learned motifs that matched known transcription factor binding sites — validating that they were capturing real biology — but they also learned patterns that had no name in any database, combinations and spacings and contextual dependencies that rule-based systems had never formalized. The field had a new instrument, and like all new instruments, it was about to reveal things that previous tools simply could not see.

---

## The Biological Challenge

For decades, molecular biologists have known that gene expression is controlled by regulatory elements scattered throughout the genome: enhancers, promoters, silencers, and insulators. These elements don't encode proteins; instead, they serve as binding platforms for transcription factors that orchestrate when, where, and how much a gene is expressed.

Here's the fundamental challenge: **the human genome contains approximately 1 million candidate regulatory regions**, but we have experimental data for perhaps 1-2% of them, and mostly in just a few cell types. Techniques like ChIP-seq (chromatin immunoprecipitation sequencing) can map where specific proteins bind, DNase-seq and ATAC-seq reveal accessible chromatin, and reporter assays measure regulatory activity. But these experiments face severe limitations:

**Cost and Scale:** A single ChIP-seq experiment costs $500-1,000 per sample. To profile just 10 histone modifications across 200 cell types would cost $1-2 million and generate 2,000 datasets to analyze.

**Time:** ChIP-seq experiments take 1-2 weeks from cell culture to sequencing results.

**Biological Material:** Many cell types are difficult or impossible to obtain in sufficient quantities—especially rare cell types, developmental stages, or samples from patients with rare conditions.

**Perturbation Testing:** Testing how individual genetic variants affect regulatory activity requires creating hundreds of cell lines or using reporter assays, each taking weeks to months.

This gap between what we want to know (regulatory activity genome-wide, across all cell types and conditions) and what we can measure creates a perfect opening for computational prediction. If we can train models on available experimental data, perhaps they can predict regulatory activity directly from DNA sequence.

---

## Learning Objectives

After completing this chapter, you will be able to:

- [ ] Explain the key types of epigenomic data (ChIP-seq, DNase-seq, ATAC-seq) and what they measure
- [ ] Articulate why experimental approaches are insufficient for genome-wide regulatory prediction
- [ ] Describe how DNA sequence can be encoded as input for neural networks
- [ ] Understand the conceptual shift from conservation-based to sequence-based prediction
- [ ] Identify what makes deep learning particularly suited for genomic sequence analysis
- [ ] Recognize the basic architecture of sequence-to-function models
- [ ] Evaluate when sequence-based prediction is appropriate versus when experiments are needed

---

## 8.1 The Epigenomic Landscape

To understand why deep learning revolutionized genomics, we first need to understand what epigenomics measures and why it matters.

### 8.1.1 What is Epigenomics?

The term "epigenome" refers to chemical modifications and structural changes to DNA and histones that affect gene expression without changing the DNA sequence itself. While your genome is essentially the same in every cell, your epigenome differs dramatically between cell types—it's why a neuron and a liver cell can have identical DNA but completely different functions.

Key epigenomic features include:

**Histone Modifications:** Histones are proteins that DNA wraps around. Chemical modifications to these proteins—methylation, acetylation, phosphorylation—mark different types of regulatory regions:
- **H3K4me3** (histone H3 lysine 4 trimethylation): marks active promoters
- **H3K27ac** (histone H3 lysine 27 acetylation): marks active enhancers
- **H3K4me1** (histone H3 lysine 4 monomethylation): marks poised or active enhancers
- **H3K27me3** (histone H3 lysine 27 trimethylation): marks repressed regions
- **H3K9me3** (histone H3 lysine 9 trimethylation): marks heterochromatin

**Chromatin Accessibility:** Open chromatin regions allow transcription factors to bind DNA. Techniques like DNase-seq and ATAC-seq identify these accessible regions by detecting where DNA is not tightly wrapped around histones.

**DNA Methylation:** Addition of methyl groups to cytosine bases, typically associated with gene repression when it occurs at promoters.

**Transcription Factor Binding:** The actual binding sites of specific proteins that regulate transcription, measured by ChIP-seq with antibodies against specific transcription factors.

### 8.1.2 Why Epigenomics Matters for Understanding Variants

Remember from Chapters 6-7 that most genetic variants don't fall in protein-coding regions. Approximately 98% of the human genome is non-coding, but this doesn't mean it's unimportant. Many variants associated with complex traits and common conditions fall in regulatory regions where they alter transcription factor binding, chromatin accessibility, or histone modifications.

For example, a single nucleotide variant might:
- Disrupt a transcription factor binding site, reducing enhancer activity
- Create a new binding site, causing inappropriate gene activation
- Alter chromatin structure, affecting accessibility of nearby regulatory elements
- Change the spacing between regulatory elements, disrupting their coordinated function

To predict the functional impact of such variants, we need to understand how DNA sequence encodes regulatory information. This is fundamentally a *sequence-to-function* problem.

### 8.1.3 The ENCODE Project and Epigenomic Data

The ENCODE (Encyclopedia of DNA Elements) Project, launched in 2003, aimed to identify all functional elements in the human genome. By 2012, ENCODE had generated thousands of datasets across multiple cell types, including:
- ChIP-seq for hundreds of transcription factors and histone modifications
- DNase-seq for chromatin accessibility
- RNA-seq for gene expression
- Hi-C for 3D chromatin structure

This massive dataset—publicly available and standardized—became the foundation for training sequence-based deep learning models. For the first time, researchers had enough labeled examples (DNA sequences with known regulatory activity) to train neural networks.

---

## 8.2 Limitations of Experimental Approaches

Let's quantify exactly why experiments alone can't solve the regulatory code.

### 8.2.1 The Combinatorial Problem

The human genome is approximately 3 billion base pairs. If we consider a typical enhancer as 200-500 base pairs, there are potentially millions of candidate regulatory regions. But the real challenge is combinatorial:

**Cell type specificity:** Regulatory activity varies dramatically across the ~200+ human cell types. An enhancer active in neurons might be silent in liver cells.

**Developmental stages:** Activity changes during development—embryonic, fetal, neonatal, adult, aging.

**Environmental conditions:** Nutrient availability, stress, signaling molecules, and disease states alter regulatory landscapes.

If we wanted to experimentally measure the activity of 1 million regulatory regions across just 200 cell types, that's 200 million experiments. At $500 per ChIP-seq experiment, the cost would be $100 billion—more than the entire Human Genome Project.

### 8.2.2 Variant Testing is Prohibitively Expensive

The average human genome contains 4-5 million single nucleotide variants compared to the reference genome. To experimentally test each variant's effect on regulatory activity would require:

1. Creating DNA constructs with and without the variant
2. Introducing them into appropriate cell types
3. Measuring changes in chromatin state, transcription factor binding, or gene expression
4. Replicating across multiple conditions

Even with high-throughput approaches like massively parallel reporter assays (MPRAs), which can test thousands of variants simultaneously, we can't test millions of variants across hundreds of cell types. MPRAs cost $10,000-50,000 per experiment.

### 8.2.3 The Need for Prediction

This creates an urgent need for **predictive models** that can:
- Predict regulatory activity directly from DNA sequence
- Generalize to unmeasured cell types and conditions
- Test millions of variants in silico before selecting candidates for experimental validation
- Provide mechanistic insights into how sequence encodes regulatory information

This is where deep learning enters the picture.

---

## 8.3 Sequence-Based Prediction: A New Paradigm

### 8.3.1 From Conservation to Sequence

In Chapters 6 and 7, we learned about tools like SIFT, PolyPhen-2, and CADD that predict variant impact using evolutionary conservation and hand-crafted features. These approaches work well for protein-coding variants but struggle with regulatory regions because:

**Conservation is weak:** Regulatory elements can be conserved in function but not sequence. Different transcription factors with similar DNA-binding preferences can compensate for each other.

**Context matters:** The same sequence motif can have opposite effects depending on surrounding sequence context.

**Combinatorial logic:** Multiple transcription factors often work together in complex ways that can't be captured by simple motif counting.

Sequence-based deep learning takes a fundamentally different approach: instead of hand-crafting features, **let the neural network learn** which sequence patterns matter for regulatory function.

### 8.3.2 The Core Insight

Here's the key conceptual leap: DNA sequence is fundamentally a discrete string of symbols (A, C, G, T), much like text is a string of characters. If neural networks can learn patterns in text (sentiment analysis, language translation), perhaps they can learn patterns in "genomic language."

The question becomes: can we train a neural network to learn the *regulatory code*—the rules that translate DNA sequence into regulatory activity?

### 8.3.3 The Sequence-to-Function Framework

The basic framework is:

**Input:** DNA sequence (e.g., 1000 base pairs)  
**Output:** Regulatory activity predictions (e.g., DNase-seq signal, histone marks, transcription factor binding)

This is a **supervised learning** problem:
1. Collect training data: sequences with known regulatory activity from experiments
2. Train a neural network to predict activity from sequence
3. Apply the trained model to new sequences

The magic is that if the model learns generalizable patterns, it can predict activity for sequences never seen experimentally.

---

## 8.4 Encoding DNA Sequence for Neural Networks

Before we can train a neural network on DNA sequences, we need to convert them into a numerical format. This process, called **encoding**, is crucial for model performance.

### 8.4.1 One-Hot Encoding

The standard approach for encoding DNA sequences is **one-hot encoding**. Each nucleotide is represented as a vector:

```
A = [1, 0, 0, 0]
C = [0, 1, 0, 0]
G = [0, 0, 1, 0]
T = [0, 0, 0, 1]
```

A DNA sequence like "ACGT" becomes a 4×4 matrix:

```
   A  C  G  T
A [1  0  0  0]
C [0  1  0  0]
G [0  0  1  0]
T [0  0  0  1]
```

For a sequence of length 1000 bp, this creates a 4×1000 matrix—the input to our neural network.

**Why one-hot encoding?**
- No implicit ordering (A isn't "less than" C)
- Treats all nucleotides equally
- Easy for networks to learn patterns
- Standard in deep learning for genomics

**Why not just encode A=1, C=2, G=3, T=4?**

A simple numerical encoding would create a false relationship—it would imply A and C are "closer" than A and G, or that the average of A and T equals C. Nucleotides don't have this kind of numeric relationship, and such a scheme would confuse the network.

### 8.4.2 Handling Ambiguous Bases

Real sequencing data sometimes contains ambiguous base calls (N for unknown). Common approaches:
- Set all four values to 0.25 (equal probability)
- Set all four values to 0 (no information)
- Mask these positions during training

### 8.4.3 Reverse Complement Awareness

DNA is double-stranded, and regulatory elements can function on either strand. Many models use **reverse complement augmentation**: during training, each sequence is randomly flipped to its reverse complement. This teaches the model that both strands contain the same information.

---

## 8.5 Why Deep Learning for Genomics?

### 8.5.1 The Right Tool for the Right Job

Several properties of genomic sequences make them ideal for deep learning:

**1. Large training datasets exist:** ENCODE, Roadmap Epigenomics, and other consortia have generated millions of labeled examples (sequence + activity measurements).

**2. Patterns are hierarchical:** Regulatory logic has natural hierarchical structure:
- Base level: individual nucleotides
- Motif level: transcription factor binding sites (6-20 bp)
- Module level: combinations of motifs (50-200 bp)
- Regional level: enhancer-promoter interactions (kilobases)

This hierarchy matches the structure of deep neural networks, where early layers detect simple patterns and deeper layers combine them into complex representations.

**3. Spatial relationships matter:** The relative positioning and spacing of motifs affects function. Convolutional neural networks (CNNs) are explicitly designed to capture spatial patterns.

**4. Long-range dependencies exist:** Regulatory elements can influence genes thousands of base pairs away. Advanced architectures can model these long-range interactions.

### 8.5.2 What Deep Learning Offers vs. Traditional Methods

Compared to the conservation-based approach (Chapter 6) and ensemble ML (Chapter 7), deep learning for genomics provides:

| Approach | Features | Best For | Key Limitation |
|----------|----------|----------|----------------|
| Conservation (Ch. 6) | Hand-crafted (PhyloP, SIFT…) | Protein-coding variants | Poor for regulatory regions |
| Ensemble ML (Ch. 7) | 63 annotations | All variant types | Hand-curated features |
| Deep Learning (Ch. 8) | Learned from raw sequence | Regulatory variants | Data-hungry; less interpretable |

**Automatic feature learning:** No need to manually define features like "number of CpG islands" or "conservation score." The network learns which sequence patterns matter.

**Multiple outputs:** Can predict multiple epigenomic marks simultaneously, learning shared representations.

**Transfer learning:** Models trained on one organism or cell type can be fine-tuned for others.

### 8.5.3 The Data Advantage

The key enabler was the availability of large-scale, standardized epigenomic datasets. ENCODE alone generated:
- ~5,000 ChIP-seq experiments
- ~400 DNase-seq experiments
- ~300 RNA-seq experiments
- Across ~100 cell types

This provided millions of training examples: genomic regions with known epigenomic states. Previous machine learning approaches were limited by small datasets; deep learning thrives on this scale.

---

## 8.6 The Basic Architecture: Sequence to Chromatin State

Let's walk through the conceptual architecture of an early sequence-to-function model.

### 8.6.1 The Overall Flow

```
Input: DNA sequence (1000 bp)
         ↓
    One-hot encoding (4 × 1000 matrix)
         ↓
    Convolutional layers (detect motifs)
         ↓
    Pooling layers (summarize patterns)
         ↓
    Fully connected layers (integrate information)
         ↓
Output: Chromatin features (e.g., DNase, H3K4me3, H3K27ac)
```

### 8.6.2 What Each Layer Does

**Convolutional layers** scan the sequence with small "filters" (typically 4-20 bp wide) to detect motifs. Each filter learns to recognize a specific sequence pattern—essentially rediscovering transcription factor binding motifs automatically. Think of each filter as a molecular "antenna" tuned to a specific sequence signal.

**Pooling layers** summarize information, reducing computational cost and helping the network focus on whether a motif is present rather than its exact position.

**Fully connected layers** integrate information across the sequence, learning how combinations of motifs determine regulatory activity.

**Output layer** produces predictions for each epigenomic feature of interest.

The key concept: the network learns a hierarchy of patterns, from simple motifs to complex regulatory logic—the same way a biologist would first identify individual transcription factor binding sites, then understand how they cooperate.

### 8.6.3 Training the Model

Training follows the standard supervised learning recipe:

1. **Prepare data:** Extract sequences from the genome (positive examples: active regulatory regions; negative examples: inactive regions)
2. **Associate labels:** For each sequence, attach experimental measurements (DNase signal, histone marks, etc.)
3. **Train:** Show the network thousands of examples, adjusting weights to minimize prediction error
4. **Validate:** Test on held-out sequences never seen during training
5. **Evaluate:** Measure how well predictions match experimental data

The result is a model that can predict regulatory activity for any DNA sequence you give it—even sequences without experimental data.

> **[Optional: The Math]**
>
> Sequence-based models frame regulatory prediction as a binary classification problem: for each genomic region, is a chromatin feature present (1) or absent (0)?
>
> The model is trained by minimizing **binary cross-entropy loss**:
>
> L = -(1/N) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
>
> Where N = number of examples, yᵢ = true label (1 if feature present, 0 if absent), and ŷᵢ = model's predicted probability. When the model correctly predicts high probability for an active region, the loss is small. When it makes a wrong prediction, the loss is large—and the model updates its internal weights to do better next time.

---

## 8.7 What Can These Models Predict?

Early sequence-based models (2015-2018) demonstrated several impressive capabilities:

### 8.7.1 Chromatin Accessibility

Given a 1000 bp DNA sequence, predict whether it will be accessible (DNase-hypersensitive or ATAC-seq positive) in specific cell types. Accuracy: 80-90% for distinguishing accessible from inaccessible regions.

### 8.7.2 Histone Modifications

Predict the presence and intensity of specific histone marks from sequence alone:
- H3K4me3 (active promoters): highly predictable (~90% accuracy)
- H3K27ac (active enhancers): moderately predictable (~80% accuracy)
- H3K27me3 (repressed regions): less predictable (~70% accuracy)

The varying predictability reflects biology: some chromatin states are more sequence-determined, others depend heavily on context and signaling.

### 8.7.3 Transcription Factor Binding

Predict binding sites for hundreds of transcription factors. This is particularly valuable because ChIP-seq for most TFs doesn't exist in most cell types.

### 8.7.4 Variant Effect Prediction

Perhaps most exciting for clinical applications: predict how genetic variants alter regulatory activity. The approach:

1. Input the reference sequence → get baseline predictions
2. Input the variant sequence → get altered predictions
3. Compare: large changes suggest functional impact

This enables **in silico mutagenesis**: systematically testing every possible single nucleotide change in a regulatory region to identify critical positions.

---

## 8.8 Case Study: Predicting Non-Coding Variant Effects

### 8.8.1 The Challenge

A genome-wide association study (GWAS) identifies a variant (rs9939609) associated with obesity and type 2 diabetes. The variant falls in an intron of the *FTO* gene—no protein sequence change. How does it affect biology?

### 8.8.2 Traditional Approach

Without computational prediction, researchers would need to:
1. Hypothesize which regulatory elements might be affected
2. Test enhancer activity using reporter assays (months of work)
3. Perform ChIP-seq in relevant cell types (expensive, requires appropriate cells)
4. Test the variant's effect in cellular models

### 8.8.3 Sequence-Based Prediction Approach

A deep learning model trained on ENCODE data can:

1. **Predict baseline activity:** Input the reference sequence around rs9939609 → predict DNase accessibility and relevant histone marks in adipocytes (fat cells)

2. **Predict variant effect:** Input the variant sequence → predict how chromatin state changes

3. **Identify mechanism:** Examine which predicted TF binding sites are disrupted

4. **Prioritize experiments:** Focus wet-lab validation on the most likely mechanism

In this real case, sequence-based models predicted that rs9939609 disrupts a binding site for the transcription factor CUX1, altering enhancer activity in preadipocytes. This prediction, made computationally in minutes, was later validated experimentally.

### 8.8.4 Impact

This approach transformed variant interpretation:
- Screen thousands of GWAS variants computationally before experiments
- Predict cell-type-specific effects without having those cell types
- Generate mechanistic hypotheses about regulatory disruption
- Prioritize variants for functional validation

---

## 8.9 Advantages and Limitations

### 8.9.1 Key Advantages

**Speed:** Predictions are nearly instantaneous (milliseconds per sequence) once the model is trained. Testing millions of variants is feasible.

**Scale:** Can predict for any genomic region, not just those with experimental data.

**Cost-effective:** Training requires computational resources but no wet-lab experiments. Predictions are essentially free.

**Hypothesis generation:** Identifies which sequence features matter, guiding mechanistic studies.

### 8.9.2 Important Limitations

**Sequence-only:** These models only see DNA sequence. They miss:
- 3D chromatin structure (enhancer-promoter loops)
- Pioneer factor binding that remodels closed chromatin
- Signaling-dependent activation
- Cell state and developmental history

**Training data bias:** Models learn patterns present in training data (mostly common cell lines). Predictions for rare cell types or disease states may be less reliable.

**Correlation vs. causation:** A model might predict that a sequence is an enhancer without understanding the mechanistic steps.

**Validation still needed:** Computational predictions generate hypotheses but don't replace experimental validation, especially for clinical applications.

### 8.9.3 When to Use Sequence-Based Prediction

**Good use cases:**
- Screening thousands of variants to prioritize for experiments
- Predicting effects in cell types where experiments are infeasible
- Exploring regulatory syntax through in silico mutagenesis
- Identifying likely transcription factor binding sites
- Generating mechanistic hypotheses

**When experiments are still necessary:**
- Final validation for clinical variant interpretation
- Testing context-dependent effects (signaling, cell state)
- Measuring quantitative effects (fold-change in expression)
- Confirming long-range interactions
- Studying dynamic processes (development, differentiation)

The best research combines both: computational predictions guide efficient experimental design, and experimental results validate and refine models.

---

## 8.10 The Road Ahead

This chapter introduced the conceptual foundation of sequence-based deep learning for regulatory genomics. We've seen why this approach emerged (experimental limitations), what it predicts (chromatin states from sequence), and how it helps (variant interpretation, hypothesis generation).

In the next chapters, we'll dive deeper:

**Chapter 9** will explore the architecture that made this possible: convolutional neural networks. You'll learn exactly how filters learn motifs, how pooling summarizes information, and how models like DeepSEA and Basenji work in detail.

**Chapter 10** will introduce transformers, a newer architecture that can capture much longer-range interactions and has led to models like Enformer that predict gene expression from megabase-scale sequences.

The journey from sequence to function is just beginning.

---

## Summary

**Key Takeaways:**

- The human genome contains ~1 million candidate regulatory regions, but experimental characterization is limited by cost ($500-1000 per experiment), time (weeks per experiment), and availability of biological material

- Experimental testing of variant effects is prohibitively expensive: testing 4-5 million variants per genome across 200+ cell types would require billions of experiments

- Sequence-based deep learning offers a paradigm shift: instead of hand-crafting features based on conservation or motifs (as in Chapters 6-7), neural networks learn regulatory patterns directly from DNA sequence

- DNA sequences are encoded as one-hot matrices (4 rows for A/C/G/T, sequence length columns) for input to neural networks — this avoids false numerical relationships between nucleotides

- The ENCODE Project generated millions of labeled training examples (sequences with measured chromatin states), enabling supervised learning at scale

- Deep learning architectures use hierarchical pattern detection: early layers learn simple motifs (6-20 bp), deeper layers learn complex regulatory logic

- Models can predict chromatin accessibility (80-90% accuracy), histone modifications (70-90% depending on mark), transcription factor binding, and variant effects on regulatory activity

- Sequence-based predictions enable in silico mutagenesis: testing millions of variants computationally before selecting candidates for experimental validation

- Key advantages include speed (milliseconds per prediction), scale (genome-wide), and hypothesis generation; key limitations include inability to capture 3D structure, signaling context, and cell state

- Best practice combines computational prediction (screening and hypothesis generation) with experimental validation (confirming mechanisms and clinical significance)

---

<details>
<summary><strong>📖 Key Terms</strong></summary>

| Term | Definition |
|------|-----------|
| **ATAC-seq (Assay for Transposase-Accessible Chromatin using sequencing)** | A technique that maps open chromatin regions genome-wide by using a transposase enzyme that preferentially inserts into accessible DNA. |
| **ChIP-seq (Chromatin Immunoprecipitation Sequencing)** | A technique that identifies where proteins (like transcription factors or modified histones) bind to DNA genome-wide. |
| **Chromatin accessibility** | The degree to which DNA is physically accessible to transcription factors and other regulatory proteins. |
| **DNase-seq (DNase I hypersensitive sites sequencing)** | A technique that maps open chromatin regions by identifying where the DNase I enzyme can cut DNA. |
| **ENCODE (Encyclopedia of DNA Elements)** | A large-scale research consortium that aims to identify all functional elements in the human genome through comprehensive experimental characterization. |
| **Enhancer** | A regulatory DNA sequence that increases transcription of genes, often from a distance, by serving as a binding platform for transcription factors. |
| **Epigenome** | The complete set of chemical modifications to DNA and histones in a cell that affect gene expression without changing the DNA sequence itself. |
| **In silico mutagenesis** | Computational testing of how sequence changes affect predictions, allowing systematic exploration of variant effects without experiments. |
| **One-hot encoding** | A method of representing DNA sequences as binary matrices where each nucleotide is encoded as a vector with a single 1 and three 0s (A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]). |
| **Regulatory elements** | DNA sequences that control when, where, and how much genes are expressed, including enhancers, promoters, silencers, and insulators. |
| **Reverse complement** | The complementary DNA sequence read in reverse direction; important because DNA is double-stranded and regulatory elements can function on either strand. |
| **Sequence-to-function models** | Neural networks that predict regulatory activity and chromatin states directly from DNA sequence. |
| **Supervised learning** | A machine learning approach where models are trained on labeled examples (input-output pairs), learning to predict outputs for new inputs. |

</details>

---

## Test Your Understanding

1. A researcher has identified 500 genetic variants associated with type 2 diabetes through GWAS. Of these, 450 fall in non-coding regions. Explain why experimental testing of all variants across relevant cell types (pancreatic beta cells, liver, muscle, adipose) is impractical, and describe how sequence-based deep learning could help prioritize variants for functional validation.

2. Consider two histone modifications: H3K4me3 (marks active promoters) and H3K27me3 (marks repressed regions). Which do you think would be more predictable from DNA sequence alone, and why? What other information might be needed to accurately predict repressed regions?

3. A sequence-based model predicts that a variant disrupts an enhancer in cardiac muscle cells. However, when tested experimentally using a reporter assay in cultured cardiomyocytes, no effect is observed. List at least three biological reasons why the experiment might not have detected an effect that truly exists in vivo.

4. Explain why one-hot encoding is used for DNA sequences rather than simpler encodings like A=1, C=2, G=3, T=4. What problems would a numerical encoding create for learning regulatory patterns?

5. A model trained on ENCODE data (mostly from immortalized cell lines) is used to predict regulatory activity in early embryonic cells. What concerns might you have about the predictions' reliability? What could be done to improve them?

6. Compare and contrast the variant effect prediction approaches of CADD (Chapter 7) and sequence-based deep learning. What are the strengths and weaknesses of each? When would you use one versus the other?

7. A variant falls in a predicted enhancer for a gene 500 kb away. The sequence-based model predicts the variant disrupts enhancer activity, but the gene's expression is unchanged in patients carrying the variant. Propose biological explanations for this discrepancy involving 3D chromatin structure.

8. The text mentions that sequence-based models can achieve 80-90% accuracy for chromatin accessibility prediction. What does the remaining 10-20% error represent biologically? What factors beyond DNA sequence determine whether a region is accessible?
