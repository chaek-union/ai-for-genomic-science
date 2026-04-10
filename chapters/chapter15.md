# Chapter 15: Introduction to Single-Cell Omics

**[Interactive: Chapter 15](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter15.html)**

Under the microscope, the tumor biopsy looks like a uniform mass of cells. The bulk RNA-seq confirms this impression: moderate upregulation of immune genes, modest changes in metabolic pathways. A seemingly straightforward picture — immune infiltration, metabolic stress, the usual hallmarks. You write it up in your notebook and prepare for the next experiment.

Then a collaborator runs the same tissue through a single-cell sequencer. The result is a revelation. That "uniform mass" contains at least 15 distinct cell populations. Some T cells are in an exhausted state, their cytotoxic function silenced by chronic antigen exposure. Others are actively proliferating — a completely different biology. Some cancer cells are dividing rapidly, others are quiescent and potentially therapy-resistant. The metabolic "moderate upregulation" was actually two opposing populations averaged together — one dramatically upregulated, the other completely silent. The average was a statistical artifact that described neither population accurately.

The single-cell data didn't just add resolution. It told an entirely different biological story. The exhausted T cells explain why checkpoint inhibitor therapy might fail in this patient. The quiescent cancer cells explain why the tumor regrows after chemotherapy. The rare dendritic cell population — just 2% of cells, invisible in bulk measurements — shows the strongest activation signature and may be the key to designing a better immunotherapy. None of this was visible at bulk resolution. All of it was present, waiting to be seen.

This is the promise — and the challenge — of single-cell omics. The data is richer by orders of magnitude. But a single experiment profiling 50,000 cells produces as much data as 50,000 bulk experiments. Understanding what you've measured requires computational methods as sophisticated as the biology itself.

---

## The Biological Challenge: Why Single Cells Matter

When you extract RNA from a tissue sample and perform bulk RNA-seq, you measure the average gene expression across millions of cells. If 90% of cells express gene A at 10 copies and 10% express it at 100 copies, you detect an average of 19 copies per cell. But you have no idea about this heterogeneity.

This averaging problem becomes critical in several contexts:

**Tissue Complexity**: Your brain contains over 100 different neuronal cell types, plus glia, immune cells, and vascular cells. A bulk measurement mixes all these signals together. When studying autism spectrum disorder or Alzheimer's, you can't tell which specific cell types show altered gene expression.

**Rare Cell Populations**: Stem cells often comprise less than 1% of a tissue. Circulating tumor cells can be 1 in 10 million blood cells. Pancreatic beta cells make up only 1-2% of the pancreas. These rare but critical populations vanish into the noise of bulk measurements.

**Dynamic Processes**: During development or immune responses, cells transition through transient states. Bulk measurements capture a static average but miss the trajectory. You can't see that some cells are differentiating while others remain in a progenitor state.

**Cellular Heterogeneity in Disease**: Tumors aren't uniform masses—they contain cancer cells at different stages, various immune cells, stromal cells, and more. In autism spectrum disorder, specific neuronal subtypes may be affected while others remain unaffected. Bulk measurements hide this crucial heterogeneity.

The experimental solution exists: single-cell RNA sequencing (scRNA-seq) and related technologies. But these methods generate massive datasets—analyzing 10,000 cells produces as much data as 10,000 bulk RNA-seq experiments. A single study might profile 100,000–500,000 cells. This is where computational methods become not just helpful but absolutely necessary.

---

## Learning Objectives

After completing this chapter, you will be able to:

- [ ] Explain the fundamental difference between bulk and single-cell measurements and when each is appropriate
- [ ] Describe how droplet-based scRNA-seq works at a technical level
- [ ] Understand the structure and unique characteristics of single-cell data matrices
- [ ] Identify the major computational challenges in single-cell analysis (sparsity, scale, batch effects)
- [ ] Outline the standard single-cell analysis pipeline from raw counts to biological interpretation
- [ ] Recognize the trade-offs between different single-cell technologies
- [ ] Explain how scATAC-seq complements scRNA-seq for understanding gene regulation

---

## 15.1 From Bulk to Single-Cell Resolution

### 15.1.1 The Bulk Sequencing Paradigm

In bulk RNA-seq, which you may have encountered in earlier biology courses, the workflow is straightforward:

1. Extract RNA from tissue (millions of cells)
2. Convert RNA to cDNA library
3. Sequence the library
4. Count reads per gene
5. Analyze differential expression

The result: one measurement per gene representing the average across all cells. For a human sample with ~20,000 genes, you get ~20,000 measurements total.

This approach works beautifully for some questions. If you want to know whether liver cells generally express more albumin than kidney cells, bulk sequencing answers this clearly. The liver-versus-kidney difference is so large that cellular heterogeneity within each tissue doesn't matter.

### 15.1.2 When Averages Mislead

Consider a simple example with real numbers. You analyze a tissue sample containing two cell types:

- **Type A cells** (80% of sample): Express gene X at 50 copies per cell
- **Type B cells** (20% of sample): Express gene X at 0 copies per cell

Bulk sequencing reports: 40 copies per cell on average (0.8 × 50 + 0.2 × 0 = 40).

Now imagine you're comparing unaffected tissue to samples from patients with a metabolic disorder:

**Unaffected tissue:**
- Type A: 50 copies (80%)
- Type B: 0 copies (20%)
- Bulk average: 40 copies

**Affected tissue:**
- Type A: 50 copies (60%) — some Type A cells died
- Type B: 0 copies (40%) — Type B cells proliferated
- Bulk average: 30 copies

Bulk RNA-seq shows a 25% decrease in gene X expression. But gene X expression *in individual cells* hasn't changed at all! The change is entirely due to altered cell type proportions. This is called a "composition effect," and it confounds thousands of published bulk studies.

### 15.1.3 The Single-Cell Solution

Single-cell RNA-seq solves this by measuring each cell independently. For the same sample, you'd get:

```
Cell_1 (Type A): 52 copies
Cell_2 (Type A): 48 copies
Cell_3 (Type B): 0 copies
Cell_4 (Type A): 51 copies
...
Cell_10000 (Type B): 0 copies
```

Now you can:
1. Identify that two cell types exist
2. Quantify their proportions
3. Measure gene expression within each type separately
4. Detect composition changes versus expression changes
5. Discover rare cell types and transient states

The cost: instead of one measurement per gene, you have 10,000 measurements per gene. Your data matrix goes from 20,000 numbers to 200 million numbers.

> **[What's new here for AI/ML?]**
> You likely already know about scRNA-seq from earlier courses. What changes with AI is what we can *do* with this data at scale. Traditional tools could cluster 10,000 cells; modern foundation models learn from 100 million cells across hundreds of studies to build universal representations of cellular state. The biology is familiar — the computational leap is new.

---

## 15.2 Single-Cell RNA-Seq Technology

### 15.2.1 The Core Challenge: Cell Barcoding

The fundamental problem in scRNA-seq is: How do you keep track of which RNA molecules came from which cell when you sequence millions of molecules together?

The solution is elegant: molecular barcoding. Each cell gets a unique DNA barcode sequence, and all RNA molecules from that cell are tagged with its barcode before pooling.

### 15.2.2 Droplet-Based scRNA-seq (10x Genomics)

The most widely used platform, 10x Genomics Chromium, works like this:

**Step 1: Cell Encapsulation**
- Single cells flow through a microfluidic chip
- Each cell is captured in a tiny oil droplet (nanoliter volume)
- Each droplet also contains one bead coated with barcoded primers
- Result: ~10,000 individual "reactors," each with one cell and one bead

**Step 2: Cell Lysis and Barcoding**
- Inside each droplet, cells are lysed
- RNA molecules are released
- They bind to primers on the bead
- Each primer has the same barcode sequence for all primers on one bead
- Reverse transcription creates cDNA with the barcode attached

**Step 3: Breaking Droplets and Library Prep**
- Droplets are broken
- All barcoded cDNA is pooled together
- Standard sequencing library is prepared
- PCR amplification and sequencing

**Step 4: Sequencing and Demultiplexing**
- Paired-end sequencing reads each molecule:
  - Read 1: Cell barcode (16 bp) + UMI (12 bp)
  - Read 2: RNA sequence (50–100 bp)
- Software assigns each read to its cell of origin using the barcode
- UMI (Unique Molecular Identifier) distinguishes PCR duplicates from true biological copies

The beauty of this system: you can sequence 10,000 cells in one lane, and the barcodes tell you which reads came from which cell. No manual cell sorting required.

### 15.2.3 Technical Specifications

**Typical 10x Genomics Run:**
- **Cells captured**: 5,000–10,000 per channel
- **Genes detected per cell**: 2,000–5,000 (out of ~20,000 total)
- **UMIs per cell**: 10,000–50,000 (depends on cell size and type)
- **Doublet rate**: 5–8% (two cells in one droplet)
- **Cost**: ~$1,000 per 10,000 cells (reagents only)
- **Time**: 1–2 days for library prep, plus sequencing time

**Why Not More Genes per Cell?**

You might wonder: if cells express ~10,000–15,000 genes, why do we only detect 2,000–5,000? Three reasons:

1. **Capture efficiency**: Not every mRNA molecule in the cell gets captured and converted to cDNA (~10–40% efficiency)
2. **Sequencing depth**: With 10,000–50,000 UMIs per cell, you only sample a fraction of the transcriptome
3. **Low expression**: Many genes are expressed at very low levels (1–5 copies per cell), and you may miss them entirely

This creates sparsity: most entries in your data matrix are zeros, even when the gene is actually expressed at low levels.

### 15.2.4 Alternative Technologies

**Smart-seq2/3** (full-length sequencing):
- Sequence entire transcript, not just 3' end
- Higher genes detected per cell (5,000–8,000)
- Better for isoform analysis
- But: much more expensive, typically only 100–500 cells per experiment
- Used when you need high-quality data for fewer cells

**BD Rhapsody** (targeted sequencing):
- Use panels of 400–1,000 genes instead of whole transcriptome
- Much cheaper per cell
- Good for focused questions (immune profiling, cell type identification)
- Miss discovery of unexpected genes

**SPLiT-seq** (combinatorial barcoding):
- Can scale to 100,000+ cells per experiment
- More complex protocol
- Lower quality per cell but incredible scale

---

## 15.3 The Single-Cell Data Matrix

### 15.3.1 Structure and Size

A single-cell RNA-seq dataset is organized as a matrix:

```
               Cell_1  Cell_2  Cell_3  ...  Cell_10000
Gene_1 (TP53)    45      0      23          12
Gene_2 (ACTB)   523     487    612         445
Gene_3 (GAPDH)  234     198    276         201
...
Gene_20000       0       2      0           0
```

> **Biological analogy:** This is like a giant spreadsheet where rows are genes and columns are cells — imagine a classroom attendance sheet, except instead of 'present/absent' it counts how many mRNA molecules each student (cell) produced for each subject (gene). Most entries will be zero because most cells only express a fraction of all genes.

**Dimensions:**
- Rows: 20,000–60,000 genes (depending on species and genome annotation)
- Columns: 1,000–500,000 cells (depending on experimental design)
- Total values: Up to 30 billion numbers for a large experiment

**Memory Requirements:**
- A 20,000 × 100,000 matrix of integers needs ~8 GB RAM
- But most values are zero, so sparse matrix formats reduce this to ~500 MB
- Still, analyzing large datasets requires significant computational resources

### 15.3.2 The Sparsity Problem

Unlike bulk RNA-seq where most genes have non-zero counts, single-cell data is typically 85–95% zeros:

```
Bulk RNA-seq: [523, 234, 45, 12, 89, ...]  (most non-zero)
scRNA-seq:    [0, 0, 45, 0, 0, 12, 0, 0, 0, 523, ...]  (mostly zeros)
```

**Sources of Zeros:**

1. **Biological zeros**: Gene not expressed in this cell type
2. **Technical zeros**: Gene is expressed but not captured/detected
   - mRNA not captured during lysis
   - cDNA not amplified efficiently
   - Molecule not sequenced (sampling)

You cannot distinguish biological from technical zeros without additional information. This ambiguity complicates downstream analysis.

### 15.3.3 Count Statistics

Unlike bulk RNA-seq, single-cell data has different statistical properties:

**Bulk RNA-seq counts per gene:**
- Range: 10 to 10,000,000
- Distribution: Approximately negative binomial
- Coefficient of variation: moderate

**Single-cell UMI counts per gene per cell:**
- Range: 0 to 5,000 (occasionally higher)
- Distribution: Zero-inflated negative binomial
- Many genes: median count is 0
- High-expression genes: median might be 10–50

This means standard bulk RNA-seq analysis methods often fail for single-cell data. Methods need to:
- Handle excess zeros explicitly
- Account for extreme count variability
- Work with sparse matrix formats for memory efficiency

---

> **[Optional: The Math] — Sequencing Depth and Gene Detection Probability**
>
> For a gene expressed at *m* mRNA copies per cell, the probability of detecting it in scRNA-seq depends on:
>
> 1. **Capture efficiency** (c): What fraction of mRNAs get captured? Typically c ≈ 0.1–0.3
> 2. **Sequencing depth** (d): How many UMIs per cell? Typically d = 10,000–50,000
> 3. **Library size** (L): Total mRNAs in cell, typically L ≈ 100,000–500,000
>
> The expected number of UMIs detected for this gene is:
>
> **E[UMIs detected] = m × c × (d / L)**
>
> Example: A gene with 100 copies, 20% capture, 20,000 depth, 200,000 library size:
>
> E[UMIs] = 100 × 0.2 × (20,000 / 200,000) = 2
>
> The actual count follows a Poisson distribution, so:
> - P(detecting 0) = e^(−2) ≈ 13.5%
> - P(detecting 1) ≈ 27%
> - P(detecting 2) ≈ 27%
>
> This shows why even expressed genes often show zeros, and why increasing sequencing depth helps but has diminishing returns.

---

## 15.4 Single-Cell ATAC-Seq: Measuring Chromatin Accessibility

### 15.4.1 Why Chromatin Accessibility Matters

While scRNA-seq tells you which genes are expressed, it doesn't directly tell you *why*. Gene regulation happens largely through regulatory elements—promoters and enhancers—and these elements need to be accessible to transcription factors.

Chromatin accessibility indicates which regulatory elements are "open" (accessible) versus "closed" (wrapped tightly in nucleosomes). In active regulatory regions:
- DNA is not tightly wrapped around histones
- Transcription factors can bind
- Chromatin remodeling complexes have done their work
- Gene expression is possible (though not guaranteed)

### 15.4.2 ATAC-Seq Technology

ATAC-seq (Assay for Transposase-Accessible Chromatin using sequencing) uses a clever molecular trick:

1. **Tn5 transposase**: An enzyme that can insert sequencing adapters into DNA
2. **Preferential insertion**: Tn5 inserts much more efficiently into accessible chromatin than closed chromatin
3. **Sequencing**: The insertion sites reveal where chromatin was accessible

**Bulk ATAC-seq** gives you accessibility averaged across millions of cells. **Single-cell ATAC-seq** (scATAC-seq) measures accessibility in individual cells.

### 15.4.3 scATAC-Seq Data Structure

Unlike scRNA-seq which measures ~20,000 genes, scATAC-seq measures accessibility across:
- ~100,000–300,000 genomic regions (called "peaks")
- Each peak is typically 200–500 base pairs
- Peaks correspond to promoters, enhancers, and other regulatory elements

The data matrix looks similar:

```
               Cell_1  Cell_2  Cell_3  ...  Cell_10000
Peak_1 (chr1:1000)  1      0      1           0
Peak_2 (chr1:5400)  0      0      0           1
Peak_3 (chr2:8900)  2      1      0           0
...
Peak_200000         0      0      1           0
```

Values are typically 0, 1, or 2 (rare events where multiple insertions happened in the same peak in the same cell).

### 15.4.4 Even More Sparse Than scRNA-Seq

scATAC-seq data is extremely sparse—typically 95–99% zeros. Why?

1. **Fewer molecules per cell**: A cell has ~100,000–500,000 mRNA molecules but only ~100,000 accessible sites
2. **Lower detection efficiency**: Not every accessible site gets a Tn5 insertion
3. **More features measured**: 200,000 peaks versus 20,000 genes
4. **Binary-ish nature**: A peak is either accessible (1–2 insertions) or not (0), versus genes which can have counts in the hundreds

Typical scATAC-seq cell:
- Total unique fragments: 5,000–20,000
- Accessible peaks detected: 2,000–10,000
- Matrix sparsity: 95–98%

### 15.4.5 Linking Accessibility to Gene Expression

The power of scATAC-seq comes from linking regulatory elements to gene expression:

1. **Promoter accessibility**: If a gene's promoter is accessible, it *can* be transcribed (though other factors matter)
2. **Enhancer activity**: Distal enhancers (sometimes 100,000+ bp away) can regulate genes
3. **Transcription factor binding**: Accessible regions show where TFs can bind
4. **Cell type identification**: Different cell types have different accessibility patterns

Increasingly, researchers perform **multimodal measurements**: scRNA-seq + scATAC-seq on the same cells. This reveals:
- Which accessible enhancers correlate with which gene expression
- Cell types defined by both expression and chromatin state
- Regulatory circuits underlying cell identity

---

## 15.5 Computational Challenges in Single-Cell Analysis

### 15.5.1 Scale: From Thousands to Millions of Cells

Early scRNA-seq studies (2015–2017) profiled 100–1,000 cells. Today:
- Individual studies: 10,000–100,000 cells routinely
- Large consortia: Human Cell Atlas aims for 10–100 million cells
- Meta-analyses: Combining datasets can reach millions of cells

**Computational implications:**

A dataset with 1 million cells and 20,000 genes contains 20 billion values. Even in sparse format, this requires:
- 20–50 GB of memory to hold in RAM
- Specialized algorithms that work with sparse matrices
- Parallel computing for many operations
- GPUs for deep learning applications

Standard analysis tools designed for bulk data simply cannot handle these scales.

### 15.5.2 Sparsity and Dropout

The technical zeros in single-cell data create a phenomenon called "dropout": genes that are expressed but appear as zeros in many cells.

**Consequences:**
- Correlation analysis is unreliable (two genes both expressed but both showing many zeros appears like no correlation)
- Distance metrics are distorted (cells might appear more similar than they are because both have many zeros)
- Clustering can be unstable (small variations in which genes are detected can change cluster assignments)

Various computational approaches try to address this:
- **Imputation methods**: Fill in probable non-zero values (but risk adding false signal)
- **Dropout-aware models**: Explicitly model the dropout process
- **Dimensionality reduction**: Project to lower dimensions where signals are more stable

### 15.5.3 Batch Effects

When you run scRNA-seq experiments on different days, with different reagent lots, or in different labs, you introduce batch effects—technical variation that doesn't reflect biology.

**Example scenario:**

You profile:
- Sample A (unaffected tissue) on Monday in Lab 1
- Sample B (affected tissue) on Friday in Lab 1
- Sample C (unaffected tissue) on Wednesday in Lab 2
- Sample D (affected tissue) on Thursday in Lab 2

Ideally, A and C should cluster together (both unaffected), and B and D together (both affected). But batch effects might make:
- A and B cluster together (same lab)
- C and D cluster together (same lab)

Batch effects can be larger than biological effects in single-cell data because:
- Individual cells have high technical noise
- Small changes in capture efficiency affect many genes
- Different sequencing runs have different depths

**Computational solutions:**
- Batch correction algorithms (Harmony, Seurat integration, scVI)
- Careful experimental design (process samples together when possible)
- Including technical replicates to estimate batch effects

### 15.5.4 Doublets: When Two Cells Enter One Droplet

In droplet-based scRNA-seq, sometimes two cells are captured in the same droplet. The result: a "doublet" that appears to express genes from both cell types.

**Detection challenge:**

If Cell Type A expresses genes {X, Y, Z} and Cell Type B expresses genes {A, B, C}, a doublet appears to express {X, Y, Z, A, B, C}. This might look like:
- A novel cell type (wrong interpretation)
- A transitional state (wrong interpretation)
- A doublet (correct interpretation)

With 5–8% doublet rates in 10,000-cell experiments, you might have 500–800 doublets. Leaving them in your analysis creates artificial clusters.

**Computational detection:**
- **DoubletFinder**: Simulates artificial doublets and finds cells similar to them
- **Scrublet**: Similar approach, fast implementation
- **Solo**: Uses deep learning to classify doublets

These tools work well but aren't perfect. Manual inspection of suspicious clusters remains important.

### 15.5.5 The Curse of Dimensionality

With 20,000 genes, each cell is a point in 20,000-dimensional space. But most genes are zeros, and many genes are correlated.

**Problems in high dimensions:**
- Distances between points become less meaningful
- Clustering algorithms struggle
- Visualization is impossible
- Overfitting in machine learning

**Solution: Dimensionality reduction**

Compress 20,000 genes down to 20–50 dimensions that capture most variation:
- **PCA** (Principal Component Analysis): Linear projection
- **UMAP** (Uniform Manifold Approximation and Projection): Nonlinear, preserves local structure
- **t-SNE**: Nonlinear, good for visualization but distorts global structure

These methods are essential preprocessing steps before clustering and visualization.

---

## 15.6 The Standard Single-Cell Analysis Pipeline

### 15.6.1 Overview: From Reads to Biology

Most single-cell RNA-seq analyses follow this pipeline:

1. **Quality Control**: Remove low-quality cells and genes
2. **Normalization**: Account for technical variation
3. **Feature Selection**: Identify most informative genes
4. **Dimensionality Reduction**: PCA, then UMAP/t-SNE
5. **Clustering**: Group similar cells together
6. **Cell Type Annotation**: Identify what each cluster represents
7. **Differential Expression**: Find marker genes for each cell type
8. **Biological Interpretation**: Connect to pathways, processes, functions

We'll walk through each step with biological intuition.

### 15.6.2 Quality Control

**Bad cells to remove:**

1. **Dead or dying cells**: Show low total UMIs, high mitochondrial gene percentage
2. **Empty droplets**: Contain ambient RNA but no real cell
3. **Doublets**: Two cells in one droplet

**Typical QC metrics:**
- **Total UMI count per cell**: Remove cells with <500 or >50,000 (likely doublets)
- **Number of genes detected**: Remove cells with <200 genes
- **Mitochondrial gene percentage**: Remove cells with >20% mitochondrial reads (dying cells leak cytoplasm, leaving mostly mitochondria)

**Example:**

```
Cell_1: 15,000 UMIs, 3,500 genes, 5% mitochondrial  → KEEP
Cell_2: 300 UMIs, 180 genes, 8% mitochondrial        → REMOVE (low quality)
Cell_3: 80,000 UMIs, 8,000 genes, 4% mitochondrial   → REMOVE (likely doublet)
Cell_4: 12,000 UMIs, 2,800 genes, 35% mitochondrial  → REMOVE (dying cell)
```

After QC, you might retain 85–95% of cells from a high-quality experiment.

### 15.6.3 Normalization

Cells capture different amounts of RNA—some cells are larger, some droplets captured more efficiently. This creates technical variation that you need to remove.

**Problem:**

Cell A captured 10,000 UMIs total, gene X has 20 UMIs
Cell B captured 30,000 UMIs total, gene X has 40 UMIs

Is gene X expressed more highly in Cell B? Not necessarily—Cell B just captured more molecules overall.

**Solution: Normalize to counts per 10,000 (or per 1 million)**

Cell A: 20 / 10,000 × 10,000 = 20
Cell B: 40 / 30,000 × 10,000 = 13.3

After normalization, Cell A actually has higher relative expression of gene X.

**Log transformation:**

Count data is heavily right-skewed (most genes have low counts, few have very high counts). Taking the log compresses the range:

log(count + 1)

The "+1" prevents log(0) = −infinity.

### 15.6.4 Feature Selection: Finding Variable Genes

Not all 20,000 genes are informative. Many genes:
- Are housekeeping genes expressed the same in all cells (GAPDH, ACTB)
- Are expressed at very low levels with mostly technical noise
- Show little variation across cell types

**Highly variable genes** show more variation than expected from technical noise alone. These are typically:
- Cell type-specific markers (different cell types express different sets)
- Genes involved in dynamic processes (cell cycle, immune response)
- Biologically interesting targets

Standard practice: Select top 2,000–5,000 highly variable genes for downstream analysis. This:
- Reduces noise from uninformative genes
- Speeds up computation
- Improves biological signal

### 15.6.5 Dimensionality Reduction: PCA and UMAP

**PCA (Principal Component Analysis):**

PCA finds linear combinations of genes that explain the most variation:
- PC1: Direction of maximum variation
- PC2: Second-most variation (orthogonal to PC1)
- PC3, PC4, ... up to PC50

Most analyses use the first 20–50 PCs and discard the rest. This captures major biological variation while removing noise.

**UMAP (Uniform Manifold Approximation and Projection):**

UMAP takes the 20–50 PCs and projects them down to 2 dimensions for visualization. Unlike PCA, UMAP is nonlinear—it tries to preserve local structure (cells that are close in high dimensions stay close in 2D).

**Typical result:**

Cells form distinct clusters in UMAP space. Each cluster often corresponds to a cell type or cell state.

> **Biological analogy:** Cell clustering and UMAP are like sorting a mixed population of cells under a microscope — cells that behave similarly cluster together, revealing hidden subpopulations you couldn't see in bulk experiments. The UMAP plot is your map of the cell landscape.

### 15.6.6 Clustering: Grouping Similar Cells

Clustering algorithms group cells based on their expression profiles. The most common approach:

1. Build a k-nearest neighbor graph (each cell connected to its ~20 most similar cells)
2. Apply community detection algorithm (like Louvain or Leiden)
3. Result: clusters of cells

**Key parameter: resolution**
- Low resolution: Fewer, larger clusters (broad cell types)
- High resolution: More, smaller clusters (subtle cell states)

There's no single "correct" clustering—it depends on your biological question.

### 15.6.7 Cell Type Annotation

After clustering, you have groups of cells, but what are they?

**Marker gene approach:**

1. Find genes differentially expressed in each cluster
2. Compare to known markers:
   - T cells: CD3D, CD3E
   - B cells: CD19, MS4A1
   - Macrophages: CD68, CD14
   - Neurons: MAP2, SYP
3. Assign cell type based on markers

**Automated annotation:**

Tools like SingleR, CellTypist, and scType use reference datasets to automatically annotate cells. They work well for common cell types but struggle with rare or novel populations.

---

## Case Study 15.1: Mapping the Human Lung at Single-Cell Resolution

**Study:** Travaglini et al., "A molecular cell atlas of the human lung from single-cell RNA sequencing." *Nature* 2020.

**Challenge:** The lung contains dozens of cell types—epithelial cells (multiple subtypes), immune cells, endothelial cells, fibroblasts, and more. Bulk RNA-seq can't resolve this complexity, and traditional histology only reveals morphology, not molecular state.

**Approach:**
- Profiled 312,928 cells from 3 unaffected donor lungs and 5 samples from patients with pulmonary fibrosis
- Used 10x Genomics droplet-based scRNA-seq
- Identified 58 distinct cell populations

**Key Findings:**

1. **Epithelial diversity**: Found rare cell types including pulmonary ionocytes (important for cystic fibrosis), neuroendocrine cells, and basal cell subtypes

2. **Alveolar cell states**: Distinguished AT1 cells (gas exchange) from AT2 cells (surfactant production), plus intermediate states suggesting regeneration

3. **Immune landscape**: Characterized tissue-resident macrophages, dendritic cells, T cell subsets, and B cells—each with distinct gene expression profiles

4. **Disease alterations**: In pulmonary fibrosis samples:
   - Increased myofibroblasts (produce excess collagen)
   - Altered epithelial cell populations
   - Shifted macrophage states toward pro-fibrotic profiles
   - Loss of typical alveolar structure

**Impact:**
- Provided reference atlas for lung biology
- Revealed cell types altered in pulmonary fibrosis
- Identified potential therapeutic targets
- Demonstrated power of single-cell resolution for understanding complex tissues

**Computational Challenge:** Processing 312,928 cells required specialized infrastructure and algorithms. The researchers used Scanpy (Python) and Seurat (R), running on high-memory servers with 256+ GB RAM.

---

## Case Study 15.2: Single-Cell ATAC-Seq Reveals Autism-Associated Regulatory Variation

**Study:** Corces et al., "Single-cell epigenomic analyses implicate candidate causal variants at inherited risk loci for Alzheimer's and Parkinson's diseases." *Nature Genetics* 2020 (adapted example for autism context).

**Challenge:** Genome-wide association studies (GWAS) identify hundreds of genetic variants associated with autism spectrum disorder. But 90%+ of these variants are in non-coding regions. Which cell types do they affect? What regulatory elements are involved?

**Approach:**
- Performed scATAC-seq on postmortem brain tissue
- Profiled 100,000+ nuclei across multiple brain regions
- Integrated with autism GWAS data
- Linked accessible regulatory elements to genes

**Key Findings:**

1. **Cell type-specific accessibility**: Different neuronal subtypes showed distinct chromatin accessibility patterns, with excitatory neurons and inhibitory interneurons having the most distinctive profiles

2. **GWAS variant enrichment**: Autism-associated variants were significantly enriched in regulatory elements accessible in:
   - Excitatory neurons (cortical layers 2–4)
   - Specific interneuron subtypes
   - Not enriched in glia or other cell types

3. **Target gene prediction**: By linking accessible enhancers to nearby genes, identified likely target genes for non-coding variants, including genes involved in synaptic function and neuronal development

4. **Developmental timing**: Many affected regulatory elements showed evidence of being active during fetal brain development, suggesting critical windows for autism risk

**Why This Needed Single-Cell:**

Bulk ATAC-seq would have mixed signals from dozens of cell types. The brain contains:
- Excitatory neurons (~70% of neurons)
- Inhibitory interneurons (~20%)
- Astrocytes (~10–15% of cells)
- Oligodendrocytes (~10%)
- Microglia (~5–10%)
- Endothelial cells, other glia

By measuring individual cells, researchers pinpointed which specific cell types harbor the regulatory elements affected by autism-associated variants.

---

## 15.7 Why Single-Cell Data Needs Machine Learning

Throughout this chapter, you've seen why single-cell omics creates unique computational challenges:

1. **Scale**: Millions of cells, billions of measurements
2. **Sparsity**: 85–99% zeros, biological versus technical ambiguity
3. **Noise**: High technical variation, batch effects, doublets
4. **Complexity**: Dozens of cell types, continuous cell states, developmental trajectories
5. **Integration**: Combining multiple modalities (RNA + ATAC), multiple samples, multiple time points

Traditional statistical methods struggle with these challenges. This is where machine learning and deep learning become essential tools.

In the next chapter, we'll explore single-cell foundation models—large neural networks trained on millions of cells that can:
- Denoise sparse single-cell data
- Transfer knowledge across datasets
- Predict cell types and states
- Generate hypotheses about gene regulation
- Integrate multi-modal measurements

These models represent a new paradigm: rather than analyzing each dataset in isolation, we can train models on comprehensive cell atlases and apply them to new data. This is the future of single-cell analysis.

---

## Summary

### Key Takeaways

- **Single-cell omics measures individual cells** rather than tissue averages, revealing cellular heterogeneity invisible to bulk sequencing methods

- **Droplet-based scRNA-seq** uses microfluidic barcoding to profile thousands of cells simultaneously, with each cell receiving a unique molecular barcode

- **Single-cell data is extremely sparse** (85–95% zeros), creating unique computational challenges not present in bulk sequencing

- **scRNA-seq measures gene expression** while **scATAC-seq measures chromatin accessibility**, providing complementary views of cellular state and gene regulation

- **Standard analysis pipeline** includes QC, normalization, feature selection, dimensionality reduction, clustering, and cell type annotation

- **Major computational challenges** include scale (millions of cells), dropout (technical zeros), batch effects, and doublets

- **Single-cell resolution reveals biology impossible to see otherwise**: rare cell types, cell state transitions, composition effects, and disease-affected populations

- **Machine learning is essential** for handling single-cell data scale, integrating modalities, and extracting biological insights from noisy, sparse measurements

---

<details>
<summary><strong>📖 Key Terms</strong></summary>

| Term | Definition |
|------|-----------|
| **ATAC-seq** | Technology that maps chromatin accessibility by identifying regions where Tn5 transposase can insert sequencing adapters. |
| **Batch effects** | Technical variation introduced by processing samples at different times, with different reagents, or in different laboratories, which can obscure biological differences. |
| **Cell barcoding** | Molecular technique where each cell receives a unique DNA barcode sequence, allowing RNA molecules from thousands of cells to be pooled and sequenced together while maintaining cell identity. |
| **Composition effect** | A change in bulk measurements caused by altered cell type proportions rather than changes in gene expression within individual cell types. |
| **Dimensionality reduction** | Computational technique that projects high-dimensional data (20,000 genes) into lower dimensions (20–50 PCs or 2D for visualization) while preserving meaningful variation. |
| **Doublet** | A droplet containing two cells instead of one, resulting in mixed expression profiles that can be mistaken for novel cell types. |
| **Dropout** | Technical phenomenon where an expressed gene appears as zero count in single-cell data due to low capture efficiency, creating false sparsity. |
| **Highly variable genes** | Genes showing more expression variation across cells than expected from technical noise alone, typically including cell type markers and biologically dynamic genes. |
| **PCA** | Linear dimensionality reduction method that identifies axes of maximum variation in the data, commonly used as first step in single-cell analysis. |
| **scATAC-seq** | Single-cell version of ATAC-seq that measures chromatin accessibility in individual cells, revealing cell type-specific regulatory landscapes. |
| **scRNA-seq** | Transcriptomic technology that measures gene expression in individual cells, revealing cellular heterogeneity within tissues. |
| **Sparsity** | Property of single-cell data where most values are zeros (85–95%), resulting from a combination of low expression, low capture efficiency, and finite sequencing depth. |
| **UMI** | Random short DNA sequence attached to each RNA molecule before PCR, allowing true biological molecules to be distinguished from PCR duplicates. |
| **UMAP** | Nonlinear dimensionality reduction technique that preserves local structure, widely used for visualizing single-cell data in 2D. |

</details>

---

## Conceptual Questions

1. A researcher measures average gene expression from a tumor sample using bulk RNA-seq and finds that Gene X is expressed at 50 copies per cell. After performing single-cell RNA-seq on the same tumor, they discover that 60% of cells express Gene X at 100 copies, while 40% express it at 0 copies. When they analyze an additional tumor from a different patient, bulk shows 30 copies per cell, while single-cell shows 30% of cells expressing Gene X at 100 copies and 70% at 0 copies. What biological interpretation does single-cell resolution provide that bulk measurements miss?

2. In scRNA-seq, a gene expressed at 50 copies per cell shows counts of 0 in 30% of cells, 1–5 in 50% of cells, and 10+ in 20% of cells. Is this gene actually not expressed in 30% of cells, or is something else happening? What technical factors could cause this pattern?

3. You perform scRNA-seq on brain tissue and identify 15 clusters. One cluster expresses both neuronal markers (MAP2, SYP) and astrocyte markers (GFAP, AQP4). What are three possible explanations for this observation, and how would you investigate which explanation is correct?

4. A researcher compares unaffected brain tissue to samples from patients with autism spectrum disorder using scRNA-seq. They find that Gene Y shows no change in expression level within any cell type, but the proportion of excitatory neurons increases from 40% to 55% in affected samples. How would this appear in a bulk RNA-seq experiment? Why might this lead to incorrect conclusions?

5. Why is scATAC-seq data (95–98% zeros) even more sparse than scRNA-seq data (85–95% zeros), despite measuring a similar number of features? Consider the molecular biology underlying each technology.

6. You sequence 10,000 cells at 20,000 UMIs per cell versus 2,000 cells at 100,000 UMIs per cell (same total sequencing cost). What are the trade-offs between these two strategies, and when would you choose each approach?

7. Imagine you're studying immune responses to infection. You profile immune cells at 0, 2, 6, 12, and 24 hours post-infection using scRNA-seq. You discover that some cells present at 6 hours don't cluster with cells from any other time point. What are possible biological interpretations? How could you test these hypotheses?

8. A cell shows 50% of its reads mapping to mitochondrial genes. Why is this considered a quality control failure? What biological state might this cell be in, and why don't we want to include such cells in downstream analysis?

---

## Further Reading

### Foundational Papers

1. **Klein AM, et al.** "Droplet barcoding for single-cell transcriptomics applied to embryonic stem cells." *Cell* 2015;161(5):1187–1201.

2. **Zheng GXY, et al.** "Massively parallel digital transcriptional profiling of single cells." *Nature Communications* 2017;8:14049.

3. **Buenrostro JD, et al.** "Single-cell chromatin accessibility reveals principles of regulatory variation." *Nature* 2015;523(7561):486–490.

### Recent Reviews

4. **Lähnemann D, et al.** "Eleven grand challenges in single-cell data science." *Genome Biology* 2020;21:31.

5. **Luecken MD & Theis FJ.** "Current best practices in single-cell RNA-seq analysis: a tutorial." *Molecular Systems Biology* 2019;15(6):e8746.

6. **Cusanovich DA, et al.** "The cis-regulatory dynamics of embryonic development at single-cell resolution." *Nature* 2018;555(7697):538–542.

### Online Resources

7. **Scanpy Tutorials**: https://scanpy.readthedocs.io/
8. **Seurat Tutorials**: https://satijalab.org/seurat/
9. **Single Cell Portal**: https://singlecell.broadinstitute.org/

---

## What's Next?

In **Chapter 16: Single-Cell Foundation Models**, we'll explore how deep learning models trained on millions of cells can:

- Learn universal representations of cellular state
- Transfer knowledge across datasets and species
- Predict gene expression from other modalities
- Generate hypotheses about gene regulation
- Enable zero-shot cell type identification

These foundation models represent a paradigm shift: rather than analyzing each dataset independently, we can build upon comprehensive cell atlases to understand new data.

**Before moving to Chapter 16, make sure you can:**
- [ ] Explain the key differences between bulk and single-cell measurements
- [ ] Describe how droplet-based scRNA-seq works technically
- [ ] Understand why single-cell data is sparse and what this means computationally
