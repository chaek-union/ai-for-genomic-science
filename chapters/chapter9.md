# Chapter 9: CNN-Based Regulatory Sequence Analysis

**[Interactive: Chapter 9](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter9.html)**

## Opening Vignette

Dr. Chen stares at her computer screen, looking at a list of 847 genetic variants identified in patients with developmental disorders. Each variant sits in a noncoding region of the genome—regions that don't code for proteins but might regulate when and where genes are turned on or off.

She needs to figure out which of these variants actually disrupt gene regulation. The traditional approach would be to test each variant using reporter assays: clone each DNA sequence into a plasmid, introduce it into cells, and measure its regulatory activity. At roughly one week and $500 per variant, testing all 847 would take 16 years and cost over $400,000. And that's just for one cell type—regulatory elements often work differently in neurons versus liver cells versus heart cells.

But here's an even bigger problem: the human genome contains roughly 1 million regulatory elements. Testing them all experimentally across even just 10 cell types would require 10 million experiments. The ENCODE Project spent over a decade and hundreds of millions of dollars mapping regulatory elements in 100+ cell types, yet they still couldn't test every possible variant in every possible condition.

This is where convolutional neural networks come in. By learning patterns from existing experimental data, these models can predict the regulatory activity of any DNA sequence—including sequences that have never been tested in a lab. They can tell Dr. Chen which of her 847 variants are likely to disrupt enhancers, silencers, or other regulatory elements, all within minutes and at essentially zero cost.

## The Biological Challenge

The noncoding genome represents about 98% of human DNA, yet we understand far less about it than we do about protein-coding regions. This vast regulatory landscape includes:

- **Promoters**: DNA sequences where transcription begins (~20,000-30,000 in the human genome)
- **Enhancers**: Sequences that boost gene expression, often located far from their target genes (~400,000-1,000,000 elements)
- **Silencers**: Sequences that repress gene expression (~50,000-100,000 elements)
- **Insulators**: Sequences that block interactions between regulatory elements (~10,000-50,000 elements)

Each regulatory element works in a cell-type-specific manner. An enhancer active in neurons might be completely inactive in muscle cells. The same DNA sequence can have different functions depending on:
- Cell type (neurons vs. hepatocytes vs. cardiomyocytes)
- Developmental stage (embryonic vs. adult)
- Environmental conditions (presence of hormones, stress signals)
- Chromatin context (which histone modifications are present)

Testing regulatory activity experimentally requires techniques like:
- **Massively Parallel Reporter Assays (MPRAs)**: Test thousands of sequences simultaneously, but still limited to tested sequences
- **ChIP-seq**: Maps where transcription factors bind, but requires specific antibodies for each protein
- **ATAC-seq/DNase-seq**: Maps open chromatin regions, but doesn't directly measure regulatory activity
- **Hi-C**: Maps 3D genome organization, but is expensive and complex

**Why we need computational approaches:**

1. **Scale**: Testing every possible single nucleotide variant in every regulatory element would require ~400 billion experiments (1 million elements × 400 bases per element × 1,000 cell types)

2. **Cost**: Even with high-throughput methods, comprehensive experimental mapping costs tens of millions of dollars per cell type

3. **Speed**: Experimental characterization takes months to years; predictions take seconds

4. **Personalization**: Each individual has ~4-5 million variants. We need to predict which variants affect regulation in their specific genome

5. **Hypothesis generation**: Computational predictions help prioritize which experiments to actually perform in the lab

The challenge is to build models that can learn the "regulatory code"—the rules that determine which DNA sequences function as enhancers, promoters, or silencers in which cell types. This is where convolutional neural networks excel: they can learn to recognize sequence patterns (motifs) and their combinations that determine regulatory function.

## Learning Objectives

By the end of this chapter, you will be able to:

- [ ] Explain how convolutional neural networks recognize sequence motifs and patterns in DNA
- [ ] Describe the architecture and capabilities of DeepSEA for predicting chromatin features from DNA sequence
- [ ] Understand how Basenji models predict gene expression across multiple cell types and conditions
- [ ] Calculate and interpret variant effect predictions using in silico mutagenesis
- [ ] Compare the strengths and limitations of different CNN architectures for genomic sequence analysis
- [ ] Evaluate when CNN-based approaches are appropriate for regulatory sequence analysis
- [ ] Apply pre-trained models to predict the functional impact of noncoding variants

## 9.1 Why CNNs for DNA Sequences?

Before diving into specific models, let's understand why convolutional neural networks are particularly well-suited for analyzing DNA sequences.

### The Pattern Recognition Problem

DNA regulatory elements work through **transcription factor binding**. Transcription factors are proteins that recognize and bind to specific DNA sequences called **motifs**. A typical motif is 6-20 base pairs long and has some flexibility—for example, the binding site for the transcription factor CTCF looks roughly like: `CCGCGNGGNGGCAG` (where N means any nucleotide).

Think of a CNN filter as a molecular scanner—much like a restriction enzyme that slides along the DNA helix looking for its specific recognition sequence before cutting. A CNN filter "slides" along the encoded DNA looking for patterns it has learned to recognize during training.

The regulatory activity of a DNA sequence depends on:
1. **Which motifs are present**: Does the sequence contain binding sites for activating or repressing transcription factors?
2. **How motifs are arranged**: Are activating motifs close together (synergy) or separated by repressive motifs?
3. **The surrounding sequence context**: Is the motif in an accessible chromatin region?

This is fundamentally a pattern recognition problem—and CNNs are excellent at recognizing patterns in sequential data.

### How CNNs Process DNA

Recall from Chapter 3 that convolutional neural networks use filters that slide across input data, detecting local patterns. For DNA sequences:

1. **DNA is encoded as a one-hot matrix**: Each nucleotide becomes a 4-dimensional vector:
   - A = [1, 0, 0, 0]
   - C = [0, 1, 0, 0]
   - G = [0, 0, 1, 0]
   - T = [0, 0, 0, 1]

2. **Convolutional filters learn motifs**: A filter of width 12 can learn to recognize a 12-bp motif. The first layer learns individual motifs (like transcription factor binding sites).

3. **Deeper layers learn motif combinations**: Subsequent layers learn how motifs are arranged relative to each other—which combinations create strong enhancers versus weak ones.

4. **Pooling captures position flexibility**: Max pooling allows the network to recognize that a motif is present without caring about its exact position within a window.

### Advantages Over Traditional Approaches

**Compared to traditional motif scanning (like FIMO or JASPAR):**

- **Learns from data**: CNNs discover relevant motifs automatically rather than requiring pre-defined motif databases
- **Captures motif combinations**: Traditional approaches score motifs independently; CNNs learn synergistic interactions
- **Handles variable spacing**: CNNs can learn that two motifs need to be 50-200 bp apart, not exactly 100 bp apart
- **Integrates weak signals**: Many enhancers work through combinations of weak motifs; CNNs can integrate these signals

**Compared to traditional machine learning (like SVMs with k-mer features):**

- **Hierarchical learning**: CNNs automatically learn a hierarchy from simple patterns to complex regulatory logic
- **Parameter efficiency**: CNNs learn hundreds of motifs with thousands of parameters; k-mer features require millions of features
- **Direct sequence input**: No need to manually engineer features like "GC content" or "number of CpG islands"

### What CNNs Learn

When we visualize the filters in trained CNNs, we find that:

**First convolutional layer**: Learns individual transcription factor motifs
- Some filters match known motifs (CTCF, AP-1, GATA, etc.)
- Others discover novel motifs not in databases
- Filters often show position weight matrices similar to experimental ChIP-seq data

**Second convolutional layer**: Learns motif pairs and spacing constraints
- Recognizes that certain transcription factors work together
- Learns optimal spacing between binding sites
- Detects repressive versus activating combinations

**Deeper layers**: Learn cell-type-specific regulatory logic
- Integrate information across longer sequence windows
- Distinguish enhancers active in different cell types
- Capture long-range dependencies (for models with large receptive fields)

This hierarchical learning mirrors the biological reality: individual transcription factors bind to motifs, combinations of transcription factors work together to regulate genes, and the cell-type-specific combination of available transcription factors determines which enhancers are active.

## 9.2 DeepSEA: Predicting Chromatin Features

**DeepSEA** (Deep learning for Sequence-based Estimation of chromatin Accessibility) was one of the first successful applications of CNNs to regulatory genomics. Published in 2015 by Jian Zhou and Olga Troyanskaya, it demonstrated that DNA sequence alone could predict chromatin features with remarkable accuracy.

Think of DeepSEA as predicting enhancer activity by training on thousands of ENCODE ChIP-seq experiments—just as a well-trained physician learns to recognize disease patterns by studying thousands of past cases, DeepSEA learned to recognize regulatory patterns by studying thousands of genomic experiments.

### The Training Data

DeepSEA was trained on data from the ENCODE Project and Roadmap Epigenomics Consortium, which includes:

- **DNase-seq**: Measures chromatin accessibility (open chromatin regions where DNA is accessible to proteins)
- **Transcription factor ChIP-seq**: Maps binding locations for 125 transcription factors
- **Histone modification ChIP-seq**: Maps histone marks like H3K4me3 (active promoters), H3K27ac (active enhancers), H3K36me3 (gene bodies), H3K27me3 (repressed regions)

The training set consisted of:
- **919 different chromatin features** (DNase + TF binding + histone marks)
- Measured across multiple cell types
- Each data point: a 1000 bp DNA sequence labeled with which features are present

### The Architecture

DeepSEA uses a relatively simple architecture:

```
Input: 1000 bp DNA sequence (4 × 1000 matrix after one-hot encoding)
    ↓
Conv Layer 1: 320 filters, width 8
    ↓ ReLU activation
    ↓ Max pooling (width 4)
    ↓
Conv Layer 2: 480 filters, width 8  
    ↓ ReLU activation
    ↓ Max pooling (width 4)
    ↓
Conv Layer 3: 960 filters, width 8
    ↓ ReLU activation
    ↓
Fully Connected Layer: 925 units
    ↓ ReLU activation
    ↓ Dropout (50%)
    ↓
Output Layer: 919 units (one per chromatin feature)
    ↓ Sigmoid activation (multi-label classification)
```

**Key design choices:**

1. **Three convolutional layers**: Captures patterns at multiple scales—individual motifs (layer 1), motif pairs (layer 2), and larger regulatory modules (layer 3)

2. **Increasing filter numbers**: More filters in deeper layers allow learning more complex patterns

3. **Small filter width (8 bp)**: Matches typical transcription factor motif lengths

4. **Multi-task learning**: Predicting 919 features simultaneously helps the network learn shared regulatory logic

5. **Sigmoid output**: Each chromatin feature is predicted independently (probability from 0 to 1)

### Performance

DeepSEA achieved impressive accuracy:

- **Area Under ROC Curve (AUC)**: Average 0.89 across all 919 features
- **Best features**: DNase-seq peaks (AUC ~0.95), common transcription factors (AUC 0.90-0.93)
- **Harder features**: Rare transcription factors in specific cell types (AUC 0.70-0.80)

**Why some features are easier than others:**
- DNase peaks mark open chromatin, which has strong sequence signatures (GC-rich, motif-dense)
- Common transcription factors (like CTCF) have clear motifs and appear in many cell types (more training data)
- Rare transcription factors have fewer training examples and may depend more on chromatin context

### Variant Effect Prediction

DeepSEA's most powerful application is predicting how genetic variants affect regulatory elements. The approach is called **in silico mutagenesis**:

1. **Get reference prediction**: Input the reference DNA sequence (1000 bp centered on the variant)
2. **Get alternative prediction**: Change the variant position to the alternative allele
3. **Calculate difference**: `Δ score = Alternative prediction - Reference prediction`

For example, consider a variant in an enhancer active in liver cells:
- Reference (G): DeepSEA predicts 0.85 probability of H3K27ac in hepatocytes
- Alternative (A): DeepSEA predicts 0.12 probability of H3K27ac in hepatocytes  
- **Δ score = -0.73**: This variant likely disrupts the enhancer in liver cells

DeepSEA computes this difference for all 919 chromatin features, giving a comprehensive view of how a variant affects regulation across cell types.

### Interpreting DeepSEA Predictions

**Strong predictions (|Δ score| > 0.5)**:
- Variant likely disrupts or creates a transcription factor binding site
- Effect is strong enough to change chromatin state
- Higher confidence in functional impact

**Moderate predictions (|Δ score| = 0.2-0.5)**:
- Variant affects binding but may not completely abolish or create it
- Effect may be cell-type-specific
- Medium confidence

**Weak predictions (|Δ score| < 0.2)**:
- Variant has minimal effect on predicted features
- May not be functionally important
- Could still be important if it affects features not in the training set

**Cell-type specificity**:
If a variant shows large Δ scores for enhancer marks (H3K27ac) specifically in brain cell types but not liver cell types, this suggests:
- The variant affects a brain-specific enhancer
- It might impact genes expressed in the brain
- It could be relevant to neurological conditions

## 9.3 Basenji: Predicting Gene Expression

While DeepSEA predicts chromatin features, **Basenji** (developed by David Kelley and colleagues at Calico, published in 2018) goes a step further: it predicts actual gene expression levels from DNA sequence.

### The Motivation

Knowing that a sequence is an enhancer (from DeepSEA) is valuable, but biologists often want to know:
- How much does this enhancer boost gene expression?
- Which gene does it regulate?
- How does expression change across different cell types?
- What happens to expression when we introduce a variant?

Basenji addresses these questions by directly predicting genome-wide gene expression and chromatin accessibility from sequence.

### The Training Data

Basenji was trained on much larger genomic windows than DeepSEA:

- **Input**: 131,072 bp DNA sequences (~131 kb, or about 20-30 genes)
- **Outputs**: 
  - Gene expression (CAGE data from FANTOM5)
  - DNase-seq tracks
  - Histone modification ChIP-seq tracks
- **Cell types**: 164 different cell types and tissues
- **Tracks**: Over 4000 separate experimental tracks

**Why such long sequences?**
- Enhancers can be located 100 kb or more from their target genes
- Need to capture long-range regulatory interactions
- Gene expression depends on multiple regulatory elements spread across large regions

### Architecture Evolution: Basenji to Basenji2

**Original Basenji (2018):**

```
Input: 131,072 bp sequence
    ↓
8 Convolutional layers (with batch norm and pooling)
    ↓
2 Dilated convolutional layers (expands receptive field)
    ↓
Dense output layer
    ↓
Output: 4229 tracks, each 1024 bins (128 bp per bin)
```

**Basenji2 (2020 update):**
- Added **residual connections** (like ResNet)
- Increased model depth to 14 convolutional layers
- Added **squeeze-and-excitation blocks** (channel attention)
- Improved handling of long-range dependencies
- Better performance, especially for long-range enhancer-gene interactions

### How Basenji Makes Predictions

Unlike DeepSEA, which makes a single prediction per sequence, Basenji makes **spatially-resolved predictions**:

1. **Input**: 131 kb DNA sequence
2. **Output**: A "track" showing predicted signal across the sequence
   - Divided into bins (128 bp per bin)
   - Each bin gets a predicted value (e.g., RNA-seq read coverage)
3. **Multiple tracks**: Separate predictions for each cell type/assay

For gene expression:
- The model predicts CAGE signal across the entire 131 kb window
- Peaks in CAGE signal indicate transcription start sites
- Peak height indicates expression level
- Can see expression of multiple genes in the window

### Variant Effect Prediction in Basenji

Basenji's variant effect prediction is more informative than DeepSEA's:

1. **Sequence-to-expression changes**: Shows exactly where in the 131 kb window the effect occurs and can identify which gene's expression changes

2. **Quantitative expression changes**: Predicts fold-change in expression (2× increase, 50% decrease, etc.), which is more directly interpretable than chromatin feature changes

3. **Cell-type-specific effects**: Can show that a variant increases expression 3-fold in heart but decreases it 50% in liver

**Example interpretation:**

Imagine a variant 50 kb upstream of the gene APOE:
- Reference: Basenji predicts expression level of 45 CAGE reads in neurons
- Alternative: Basenji predicts expression level of 12 CAGE reads in neurons
- **Effect**: 73% reduction in neuronal APOE expression

This is much more directly interpretable than "0.5 change in H3K27ac prediction"—we can immediately see this variant might affect neuronal function by reducing APOE expression.

### Basenji's Performance

Basenji achieved:
- **Cross-species conservation**: Trained on mouse and human data, successfully predicts in both species
- **Gene expression correlation**: r = 0.78 between predicted and observed expression (across test genes)
- **Variant validation**: Successfully predicts direction of expression change for experimentally validated variants

**Limitations:**
- Still can't predict context that isn't in the sequence (e.g., effect of distant enhancers beyond 131 kb)
- Requires substantial computational resources (large models, long sequences)
- Less accurate for genes with very low expression
- Cannot predict effects of trans-acting factors (proteins that regulate from different chromosomes)

## 9.4 Comparing DeepSEA and Basenji

Both models use CNNs to predict regulatory function from sequence, but they differ in important ways:

| Aspect | DeepSEA | Basenji |
|--------|---------|---------|
| **Input size** | 1,000 bp | 131,072 bp |
| **Primary output** | Chromatin features | Gene expression + chromatin |
| **Spatial resolution** | Single value per feature | Track across sequence (128 bp bins) |
| **Number of tracks** | 919 | 4000+ |
| **Computational cost** | Low (fast predictions) | High (slower, more memory) |
| **Best use case** | Variant prioritization, TF binding | Expression changes, long-range effects |
| **Can identify target gene** | No (too short) | Yes (if within 131 kb) |

**When to use each:**

**Use DeepSEA when:**
- You want to quickly screen thousands of variants
- You need to know which specific transcription factors are affected
- You're studying variants in known regulatory elements
- Computational resources are limited

**Use Basenji when:**
- You want to predict quantitative expression changes
- You need to identify which gene is affected
- You're studying long-range regulatory interactions
- You have access to substantial computational resources (GPUs)

**Use both when:**
- You want comprehensive regulatory analysis
- DeepSEA identifies likely regulatory variants → Basenji predicts expression impact
- Cross-validation of predictions increases confidence

> **[선택: 수식으로 보면]**
>
> **Variant Effect Score (변이 효과 점수)**
>
> The basic formula for variant effect prediction:
>
> $$\Delta S_f = S_f(\text{alt}) - S_f(\text{ref})$$
>
> Where $\Delta S_f$ = change in score for feature $f$, $S_f(\text{alt})$ = prediction for alternative allele, $S_f(\text{ref})$ = prediction for reference allele.
>
> **Example Calculation**
>
> Reference sequence (G at position 500):
> - H3K27ac in hepatocytes: 0.82
> - H3K4me1 in hepatocytes: 0.91
> - DNase in hepatocytes: 0.88
>
> Alternative sequence (A at position 500):
> - H3K27ac in hepatocytes: 0.23
> - H3K4me1 in hepatocytes: 0.85
> - DNase in hepatocytes: 0.31
>
> Variant effects:
> - $\Delta S_{\text{H3K27ac}} = 0.23 - 0.82 = -0.59$ ← Strong negative effect
> - $\Delta S_{\text{H3K4me1}} = 0.85 - 0.91 = -0.06$ ← Minimal effect
> - $\Delta S_{\text{DNase}} = 0.31 - 0.88 = -0.57$ ← Strong negative effect
>
> **Biological interpretation**: The variant likely disrupts an active enhancer (H3K27ac down), while the enhancer remains in a poised state (H3K4me1 unchanged), and chromatin becomes less accessible (DNase down).
>
> For overall impact, we can also aggregate across features:
> $$\text{Impact Score} = \sum_{f \in \text{relevant features}} |\Delta S_f| \times w_f$$
> where $w_f$ is a weight for feature importance (e.g., weighting H3K27ac heavily for enhancers).

## 9.5 Architecture Innovations

The success of DeepSEA and Basenji inspired many architectural improvements.

### Dilated Convolutions

**Problem**: Standard convolutions have limited receptive fields. To see 1000 bp, you'd need many layers of 3-bp filters.

**Solution**: **Dilated convolutions** (also called atrous convolutions) insert gaps between filter positions.

A standard 3-bp filter looks at positions: [i, i+1, i+2]
A dilated filter with dilation=2: [i, i+2, i+4]
With dilation=4: [i, i+4, i+8]

**Advantages:**
- Exponentially increasing receptive field without adding parameters
- Can see long-range patterns (e.g., motifs 100 bp apart)
- Maintains resolution (no information loss from pooling)

### Residual Connections

**Problem**: Very deep networks are hard to train (vanishing gradients)

**Solution**: **Residual connections** (from ResNet) allow gradients to flow directly through the network.

```
x → Conv → ReLU → Conv → Add → ReLU → ...
↓                          ↑
└─────── skip connection ──┘
```

**Advantages:**
- Enables training networks with 50+ layers
- Each layer learns refinements rather than complete transformations
- Improves gradient flow during training

### Multi-Scale Feature Extraction

**Problem**: Regulatory patterns exist at multiple scales (6 bp motifs, 50 bp motif pairs, 500 bp regulatory modules)

**Solution**: Use multiple parallel convolutional paths with different filter sizes, then combine.

```
Input → Conv(width=4) → Concat → ...
     ↘ Conv(width=8) ↗
     ↘ Conv(width=16) ↗
```

### Attention Mechanisms

**Problem**: Not all parts of a sequence are equally important

**Solution**: **Attention mechanisms** learn to focus on important regions. In a genomics context, attention might focus on transcription factor binding sites while ignoring repetitive elements. (We explore this fully in Chapter 10.)

## 9.6 Case Study: Variant in TAL1 Enhancer

Let's walk through a real example of using CNN models for variant interpretation.

### Background

**TAL1** (T-cell acute lymphocytic leukemia 1) is a transcription factor critical for blood cell development. A +51 kb enhancer upstream of TAL1 is active specifically in erythroid cells (red blood cell precursors).

In 2015, researchers identified a single nucleotide variant in this enhancer:
- Position: chr1:47,690,516 (hg19)
- Reference allele: C
- Alternative allele: T
- Found in: A patient with altered red blood cell development

### Experimental Validation

The researchers performed reporter assays:
- **Reference (C)**: Strong enhancer activity in erythroid cells
- **Alternative (T)**: 78% reduction in enhancer activity
- **Mechanism**: The variant disrupts binding of the transcription factor GATA1

This took approximately 6 months of lab work and cost ~$15,000.

### DeepSEA Predictions

Running the variant through DeepSEA (this takes ~30 seconds):

**Top predicted effects:**
1. GATA1 binding in K562 cells: Δ score = -0.71
2. GATA2 binding in K562 cells: Δ score = -0.68
3. H3K27ac in K562 cells: Δ score = -0.52
4. DNase in K562 cells: Δ score = -0.61

(K562 is an erythroid cell line commonly used in ENCODE)

**Interpretation:**
- DeepSEA correctly predicts disruption of GATA1 binding
- Also predicts GATA2 disruption (makes sense—GATA factors have similar motifs)
- Predicts reduced enhancer activity (H3K27ac) and accessibility (DNase)
- Predictions are specific to erythroid cells (minimal effects in other cell types)

**Accuracy:** DeepSEA's predictions matched the experimental results closely. The model predicted the variant would disrupt GATA binding and reduce enhancer activity—exactly what was observed.

### Basenji Predictions

Running through Basenji2 (takes ~2 minutes on GPU):

**Predicted expression changes for TAL1:**
- Reference: 125 CAGE reads per bin in erythroid cells
- Alternative: 38 CAGE reads per bin in erythroid cells
- **Effect: 70% reduction in predicted TAL1 expression**

**Interpretation:**
- Basenji predicts substantial reduction in TAL1 expression
- Effect is specific to erythroid lineage cells
- Quantitative prediction (70% reduction) is close to experimental observation (78% reduction)

### Lessons from This Case

1. **Speed**: Computational predictions in minutes vs. months of experiments
2. **Accuracy**: Both models correctly predicted the functional effect
3. **Mechanism**: DeepSEA identified the specific transcription factor affected
4. **Quantification**: Basenji predicted magnitude of expression change
5. **Specificity**: Both correctly predicted erythroid-specific effect

**When models disagree with experiments:**
- Model might be wrong (limited by training data)
- Experiment might have artifacts (reporter assays don't capture all biology)
- True effect might depend on factors not in the sequence (e.g., distant enhancers, trans-acting factors)

## 9.7 Limitations and Challenges

While CNN-based models have been remarkably successful, they have important limitations.

### What Sequence Models Cannot Capture

**1. Trans-acting factors:**
Models only see DNA sequence. They can't predict whether a transcription factor is expressed in a given cell, whether it is post-translationally modified, or whether signaling pathways have activated or repressed it.

**2. Long-range interactions:**
Even Basenji's 131 kb window can't capture enhancers located 500 kb or more from target genes, inter-chromosomal interactions, or topologically associating domain (TAD) structures.

**3. DNA methylation:**
Most models don't include information about CpG island methylation status, parent-of-origin effects (imprinting), or age-related methylation changes.

**4. Chromatin accessibility context:**
Models predict from sequence alone, but real regulatory activity depends on whether chromatin is already open in that cell type and nucleosome positioning.

**5. Environmental and developmental context:**
Models can't predict how regulation changes with hormonal signals, stress responses, developmental timing, or disease states.

### Training Data Biases

- Most training data comes from immortalized cell lines (K562, HepG2, etc.)
- Models perform best on cell types similar to training data
- Training data is biased toward gene-rich, accessible chromatin regions
- Models may perform poorly in heterochromatin or repetitive regions

### When NOT to Trust Model Predictions

Be skeptical of predictions when:

1. The variant is in a poorly studied region with no ENCODE data
2. The cell type is not represented in training data
3. The variant is in a repetitive region (Alu, LINE elements)
4. The prediction is near threshold (|Δ score| < 0.2)
5. Multiple models give contradictory predictions

## Summary

### Key Takeaways

- **CNNs are natural for DNA sequence analysis** because they detect local patterns (motifs) and learn hierarchical representations (motifs → motif combinations → regulatory logic)

- **DeepSEA predicts chromatin features** from 1000 bp sequences, including transcription factor binding and histone modifications across 919 features and multiple cell types

- **Basenji predicts gene expression** from 131 kb sequences, enabling quantitative predictions of expression changes and identification of affected genes

- **Variant effect prediction** works through in silico mutagenesis: comparing model predictions between reference and alternative alleles to calculate Δ scores

- **Architecture innovations** like dilated convolutions, residual connections, and attention mechanisms have improved model performance and interpretability

- **CNN models have limitations**: they only see sequence, miss long-range interactions, depend on training data coverage, and can't capture trans-acting factors or environmental context

- **Real-world success**: Models like DeepSEA and Basenji have successfully predicted functional variants, guided experiments, and accelerated regulatory genomics research

<details>
<summary><strong>📖 Key Terms</strong></summary>

| Term | Definition |
|------|-----------|
| **Attention mechanism** | Neural network component that learns to weight different parts of input differently, focusing on important regions |
| **Basenji** | CNN-based model that predicts gene expression and chromatin accessibility from DNA sequences up to 131 kb long |
| **CAGE (Cap Analysis of Gene Expression)** | Technique that identifies transcription start sites and measures expression levels |
| **Chromatin feature** | Experimentally measurable property of chromatin, such as DNase sensitivity, histone modifications, or transcription factor binding |
| **Convolutional filter** | Learnable pattern detector that slides across DNA sequence, typically learning transcription factor motifs |
| **DeepSEA** | CNN-based model that predicts 919 chromatin features from 1000 bp DNA sequences |
| **Dilated convolution** | Convolutional operation with gaps between filter positions, allowing larger receptive fields without additional parameters |
| **Enhancer** | Regulatory DNA sequence that increases gene expression, often located far from target genes and active in specific cell types |
| **In silico mutagenesis** | Computational technique for predicting variant effects by comparing model predictions between reference and alternative sequences |
| **Multi-task learning** | Training approach where model predicts multiple related outputs simultaneously, learning shared representations |
| **One-hot encoding** | Representation of DNA where each nucleotide becomes a 4-dimensional binary vector (A, C, G, T) |
| **Receptive field** | Region of input sequence that influences a particular output, determined by filter sizes and number of layers |
| **Regulatory element** | DNA sequence that controls gene expression without coding for protein, including enhancers, promoters, silencers, and insulators |
| **Residual connection** | Skip connection that adds layer input to output, enabling training of very deep networks |
| **Variant effect prediction** | Computational estimation of how a genetic variant affects regulatory function or gene expression |

</details>

## Test Your Understanding

1. Why are convolutional neural networks particularly well-suited for analyzing DNA sequences, compared to fully connected networks? What properties of CNN architecture match the biological properties of regulatory elements?

2. DeepSEA was trained on 919 chromatin features from multiple cell types. Explain how multi-task learning (predicting all features simultaneously) helps the model learn better representations than training 919 separate models.

3. A researcher finds that DeepSEA predicts a variant disrupts GATA1 binding (Δ score = -0.82) in K562 cells but has minimal effect on GATA1 binding in HepG2 cells (Δ score = -0.05). What might explain this cell-type-specific prediction? What additional information would help interpret this result?

4. Basenji can predict gene expression from sequence, but it cannot predict how expression changes when a signaling pathway is activated (e.g., when cells receive a hormone signal). Explain what information Basenji is missing and why this limitation exists.

5. Compare the advantages and disadvantages of DeepSEA's 1 kb input window versus Basenji's 131 kb window. For what types of biological questions would each be more appropriate?

6. A variant shows a strong DeepSEA prediction (disrupts H3K27ac, Δ score = -0.67) but weak Basenji prediction (minimal expression change). Propose three biological explanations for why these predictions might disagree.

7. Imagine you're studying a rare neurodevelopmental condition and have identified 200 noncoding variants in affected individuals. Design a computational pipeline using DeepSEA and Basenji to prioritize which variants to validate experimentally. What criteria would you use?

8. CNN models can learn transcription factor motifs from data without being given a motif database. How would you validate that a learned filter actually represents a biologically meaningful motif? What experiments or analyses would you perform?
