# Chapter 16: Single-Cell Foundation Models

**[Interactive: Chapter 16](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter16.html)**

## Opening Vignette

Dr. Chen has just received scRNA-seq data from 50,000 cells isolated from pancreatic tissue samples collected from patients with type 2 diabetes and unaffected controls. She needs to identify which cells are insulin-producing beta cells, which are in a stressed state, and which show expression patterns associated with the disorder.

Her graduate student spent three weeks manually annotating 2,000 cells using known marker genes. But the annotations are inconsistent—some cells express both beta cell markers and alpha cell markers. Others show inflammation signatures that don't fit standard categories. And there are dozens of rare cell populations that don't match anything in the literature.

What if she could use a model trained on millions of cells from hundreds of studies? A model that has learned the "language" of gene expression across tissues, conditions, and species? A model that could automatically identify cell types, predict cellular responses to perturbations, and even suggest which genes to target for therapeutic intervention?

This is the promise of single-cell foundation models—AI systems trained on vast amounts of single-cell data that can be applied to new biological questions without retraining from scratch.

---

## The Biological Challenge

Single-cell RNA-sequencing has generated an unprecedented explosion of data. As of 2024, public repositories contain gene expression measurements from over 100 million individual cells. The Human Cell Atlas alone aims to profile every cell type in the human body—an estimated 37 trillion cells across 200+ major cell types.

But this wealth of data creates new challenges:

**The Scale Problem**: Each scRNA-seq experiment generates a matrix with 20,000–30,000 genes (rows) and 10,000–500,000 cells (columns). That's up to 15 billion measurements per experiment. Training a model on even a fraction of published datasets requires processing tens of billions of data points.

**The Integration Problem**: Cells sequenced in different labs, using different protocols, from different individuals show substantial technical variation. A beta cell from Lab A looks different from a beta cell from Lab B, even though they're biologically similar. Integrating data across studies requires sophisticated methods to separate biological variation from technical noise.

**The Annotation Problem**: Manually identifying cell types requires expert knowledge and is incredibly time-consuming. The same cell type may have different names in different papers (pancreatic beta cell = β-cell = insulin-producing cell = INS+ cell). Creating consistent annotations across millions of cells is impossible without automation.

**The Generalization Problem**: Traditional machine learning models trained on one tissue or condition often fail when applied to new contexts. A model trained on healthy pancreatic tissue may not recognize stressed beta cells from patients with diabetes. We need models that capture fundamental principles of cellular biology that transfer across contexts.

---

## Learning Objectives

By the end of this chapter, you should be able to:

- [ ] Explain how transfer learning applies to single-cell data analysis
- [ ] Describe how transformer architectures are adapted for gene expression data
- [ ] Compare and contrast different single-cell foundation models (scBERT, Geneformer, scGPT)
- [ ] Understand how these models create meaningful gene and cell embeddings
- [ ] Evaluate the appropriate use cases for different foundation model approaches
- [ ] Interpret model predictions in terms of biological mechanisms

---

## 16.1 The Foundation Model Paradigm

### What Makes a Model "Foundational"?

The term "foundation model" emerged in 2021 to describe large AI systems trained on broad data that can be adapted to many downstream tasks. Think of GPT-3 for language or CLIP for images. These models learn general representations during pre-training, then can be fine-tuned for specific applications with minimal additional data.

For single-cell biology, this paradigm is transformative. Instead of training a new neural network from scratch for every biological question, we can:

1. **Pre-train** once on massive datasets (millions of cells)
2. **Fine-tune** quickly for specific tasks (hours instead of days)
3. **Apply** zero-shot or few-shot learning to new cell types

The key innovation is that the pre-trained model learns a "universal language" of gene expression—patterns that hold across tissues, conditions, and even species.

### Why Single-Cell Data is Perfect for Foundation Models

Single-cell datasets share critical properties with language:

**Structure**: Just as words appear in sequences with grammatical rules, genes are expressed in coordinated patterns governed by regulatory networks. Co-expressed genes are like phrases; regulatory modules are like grammar.

**Context-dependence**: A gene's "meaning" depends on what other genes are expressed, just as a word's meaning depends on surrounding words. The gene CD4 in a T cell context has different biological implications than CD4 in a different cellular context.

**Transferability**: Core biological principles (cell cycle, stress response, differentiation) appear across cell types, similar to how narrative structure transfers across texts.

**Scale**: We have hundreds of millions of single-cell profiles—sufficient data to learn meaningful patterns.

> **Biological analogy:** Think of scBERT, Geneformer, and scGPT as BERT for text, but instead of words, the "vocabulary" is genes, and instead of sentences, the "text" is a cell's gene expression profile. Just as BERT learns that "bank" means something different in "river bank" vs. "bank account," these models learn that CD4 means something different in a T cell versus a macrophage.

### The Pre-training Objective

Most single-cell foundation models use **masked gene prediction** as their pre-training task, directly inspired by BERT's masked language modeling:

1. Start with a cell's gene expression profile (20,000 genes)
2. Randomly mask 15% of gene values
3. Train the model to predict the masked genes from the unmasked ones
4. The model learns gene-gene relationships and cellular state representations

This seemingly simple task forces the model to learn:
- Which genes are co-regulated
- Which gene expression patterns define cell types
- How gene expression changes during cellular processes
- What "normal" gene expression looks like (establishing a reference for detecting altered states)

---

## 16.2 scBERT: Adapting BERT to Single Cells

### Architecture and Design

scBERT (2022) was one of the first attempts to apply transformer architecture to single-cell data. The key innovation was treating genes as "words" and cells as "sentences."

**Input representation**: Each gene's expression is represented as:
```
Gene Embedding = Gene Identity Embedding + Expression Value Embedding
```

The gene identity tells the model "this is the BRCA1 gene," while the expression value indicates "it's highly expressed" or "barely detectable." This dual embedding captures both what genes are present and how active they are.

**Architecture specifications**:
- 12 transformer layers
- 128-dimensional embeddings
- 8 attention heads per layer
- Trained on 10 million cells from 66 datasets
- Pre-training took ~200 GPU hours

### How scBERT Processes Cells

Let's walk through what happens when scBERT sees a T cell:

1. **Tokenization**: The cell's 20,000 gene expression values are converted to tokens. Genes with zero or very low expression might be filtered out, leaving ~5,000–8,000 "active" genes.

2. **Embedding**: Each gene gets a 128-dimensional vector that combines gene identity and expression level. Highly variable genes (like T cell markers CD3D, CD3E) get specialized representations.

3. **Self-attention**: The transformer layers allow every gene to "attend" to every other gene. CD3D can look at CD3E, CD8A, and other T cell markers to understand the cellular context.

4. **Cell-level representation**: The final hidden states are pooled to create a single vector representing the entire cell's transcriptional state.

### Practical Applications

**Cell type annotation**: After pre-training, scBERT can be fine-tuned on labeled cells from one dataset and achieve >95% accuracy on held-out test cells. When applied to completely new datasets (zero-shot), it still achieves ~85% accuracy.

**Batch effect correction**: Because scBERT learned gene expression patterns across many studies, it naturally learns to separate biological variation (real differences between cell types) from technical variation (differences between sequencing runs).

**Gene imputation**: scBERT can predict expression levels for genes that were missed by sequencing (dropouts). The model uses patterns from other genes to infer what the missing values should be.

### Limitations

scBERT treats all genes equally, but we know that transcription factors and signaling molecules have outsized importance in determining cell state. The model doesn't incorporate our prior biological knowledge about gene regulatory networks.

Additionally, scBERT operates on processed count data, losing information about RNA splicing, velocity (directionality of change), or spatial context that might be available in the original data.

---

## 16.3 Geneformer: Gene Networks as Context

### A Network-Aware Design

Geneformer (2023) took a different approach by incorporating prior knowledge about gene function. Instead of treating all genes as equal tokens, Geneformer ranks genes by their importance in each cell.

**Rank-value encoding**: Genes are sorted by expression level within each cell:
- Rank 1: Highest expressed gene
- Rank 2: Second highest
- ...
- Rank N: Lowest detected gene

This rank-based approach makes the model robust to technical variation in absolute expression levels. A gene might have 100 counts in one experiment and 1,000 counts in another, but if it's the 5th most highly expressed gene in both cases, it gets the same rank.

**Why ranking matters**: In single-cell data, the relative order of gene expression often matters more than absolute values. If FOXP3 is among the top 50 expressed genes in a T cell, that's probably a regulatory T cell—regardless of whether FOXP3's count is 50 or 500.

### The Geneformer Architecture

**Model specifications**:
- 6 transformer layers (smaller than scBERT)
- Pre-trained on 30 million cells from 250+ datasets
- Considers top 2,048 genes per cell
- Uses gene embeddings from Gene Ontology annotations
- Can process cells in ~10 milliseconds

**Integration with biological knowledge**: Geneformer initializes gene embeddings using Gene Ontology (GO) annotations. Genes with similar functions (e.g., all DNA repair genes) start with similar embeddings. This gives the model a "head start" based on decades of biological knowledge.

### In-Context Learning

Perhaps Geneformer's most impressive capability is **in-context learning**—the model can perform tasks without any fine-tuning, just by seeing examples.

**Example**: Identifying cardiomyocytes
1. Show Geneformer 10 example cardiomyocyte cells
2. Give it a new unlabeled cell
3. The model predicts whether the new cell is also a cardiomyocyte

The model accomplishes this by comparing the new cell's gene expression pattern to the examples, using patterns learned during pre-training. This is analogous to how GPT-3 can translate languages it wasn't explicitly trained to translate—it learned general principles that transfer.

### Mechanistic Interpretations

Because Geneformer uses attention mechanisms, we can examine which genes the model focuses on when making predictions. When classifying heart cells, the model attends most strongly to cardiac transcription factors (GATA4, NKX2-5, MEF2C)—exactly the genes a human expert would look at.

This interpretability is crucial for biological applications. We don't just want a "black box" that makes good predictions; we want to understand why it makes those predictions and whether its reasoning aligns with biological mechanisms.

---

## 16.4 scGPT: Generative Pre-training for Cells

### From Classification to Generation

scGPT (2023) asks a different question: instead of just predicting masked genes, can we generate entirely new cellular states? This generative approach enables new applications like:

- Predicting how cells respond to drugs
- Simulating cellular differentiation trajectories
- Generating synthetic training data for rare cell types

**Generative modeling**: Given a cell's current state, scGPT learns the probability distribution over possible next states. This is similar to how GPT predicts the next word in a sentence, but here we're predicting cellular state transitions.

### Architecture Innovations

**Multi-task pre-training**: scGPT trains on three tasks simultaneously:

1. **Masked gene prediction** (like scBERT)
2. **Cell-cell similarity prediction** (which cells are neighbors in the tissue?)
3. **Generative modeling** (what genes should be expressed together?)

This multi-task approach forces the model to learn different aspects of cellular biology:
- Task 1: Gene co-expression patterns
- Task 2: Spatial and temporal relationships
- Task 3: Causal structure (what causes what)

**Model specifications**:
- 12 transformer layers
- 512-dimensional embeddings
- Pre-trained on 33 million cells
- Can condition on cell type, tissue, and perturbation information
- Processes ~100 cells per second

### Conditional Generation

A key innovation in scGPT is **conditional generation**—the ability to generate cells with specific properties.

**Example**: Simulating drug treatment
```
Input: Baseline cardiomyocyte + "doxorubicin treatment"
Output: Predicted gene expression after drug exposure
```

The model learned patterns of how cells respond to perturbations by training on datasets where the same cell types were measured with and without treatments. It can then predict responses to drugs or genetic perturbations it hasn't seen during training.

### Perturbation Prediction

One of scGPT's most impressive capabilities is predicting cellular responses to genetic perturbations:

**Task**: What happens if we knock out gene X in cell type Y?

The model approaches this by:
1. Taking a reference cell of type Y
2. Setting gene X's expression to zero
3. Propagating this change through the learned gene network
4. Predicting compensatory changes in other genes

For example, knocking out a transcription factor might cause downstream target genes to decrease, while stress response genes increase to compensate. The model predicts these cascade effects.

**Validation**: When tested on CRISPR knockout experiments, scGPT's predictions correlated 0.7–0.8 with actual measured expression changes—not perfect, but remarkably good considering the complexity of cellular responses.

---

## 16.5 Model Comparison and Selection

### Architectural Differences

| Model | Released | Cells | Parameters | Key Innovation | Best For |
|-------|----------|-------|------------|----------------|----------|
| **scBERT** | 2022 | 10M | 15M | First BERT for scRNA-seq | Cell type annotation |
| **Geneformer** | 2023 | 30M | 10M | Rank-value encoding | Zero-shot learning |
| **scGPT** | 2023 | 33M | 52M | Generative modeling | Perturbation prediction |
| **scFoundation** | 2024 | 50M | 100M | Multi-modal (RNA+ATAC) | Integration tasks |
| **CellPatch** | 2024 | 45M | 80M | Spatial context | Tissue architecture |

### When to Use Each Model

**Use scBERT when**:
- You have standard scRNA-seq data
- You need fast, accurate cell type annotation
- You're integrating data across batches
- Interpretability is less critical than accuracy

**Use Geneformer when**:
- You have limited labeled training data
- You need to identify rare cell types
- You want biological interpretability
- You're working with new cell types not in your training data

**Use scGPT when**:
- You're predicting responses to perturbations
- You need to generate synthetic cells
- You're studying cellular dynamics
- You have paired perturbation datasets

**Use multi-modal models (scFoundation) when**:
- You have both RNA and chromatin accessibility data
- You're studying gene regulation
- You need to integrate across data modalities

### Performance Benchmarks

On standard cell type annotation tasks across 20 datasets:
- scBERT: 94.2% accuracy (±2.1%)
- Geneformer: 93.8% accuracy (±2.4%)
- scGPT: 95.1% accuracy (±1.8%)

But these numbers don't tell the full story. Geneformer excels in zero-shot scenarios (new cell types), while scGPT is best for perturbation prediction. Choose based on your biological question, not just overall accuracy.

---

> **[선택: 수식으로 보면] — Attention Mechanisms in Single-Cell Models**
>
> The self-attention mechanism is central to all these foundation models. Here's how it works for gene expression:
>
> **Input**: A cell with gene expression values g₁, g₂, ..., gₙ
>
> **Step 1: Create Query, Key, Value matrices**
> ```
> Q = W_Q × [g₁, g₂, ..., gₙ]
> K = W_K × [g₁, g₂, ..., gₙ]
> V = W_V × [g₁, g₂, ..., gₙ]
> ```
> Where W_Q, W_K, W_V are learned weight matrices.
>
> **Step 2: Compute attention scores**
> ```
> Attention(gᵢ, gⱼ) = exp(Qᵢ · Kⱼ / √d) / Σⱼ exp(Qᵢ · Kⱼ / √d)
> ```
> This score represents "how much should gene i pay attention to gene j?"
>
> **Step 3: Weighted combination**
> ```
> Output_i = Σⱼ Attention(gᵢ, gⱼ) × Vⱼ
> ```
>
> **Biological interpretation**: High attention from CD3D to CD3E makes sense—they're both T cell receptor components. High attention from FOXP3 to IL2RA makes sense—both are regulatory T cell markers. The model learns these gene-gene relationships from data.

---

## 16.6 Fine-Tuning for Specific Tasks

### The Transfer Learning Pipeline

Foundation models are powerful, but you'll almost always need to adapt them to your specific biological question. Here's the typical workflow:

**Step 1: Load pre-trained model** — obtain the publicly released model weights from the authors' repository.

**Step 2: Prepare your data** — normalize and filter your scRNA-seq data as described in Chapter 15.

**Step 3: Fine-tune on labeled subset** — use 1,000 labeled cells to teach the model cell type classification for your specific tissue.

**Step 4: Apply to all cells** — classify all 50,000 cells using the fine-tuned model.

### How Much Data Do You Need?

This is where foundation models shine. Traditional approaches might need:
- 10,000+ labeled cells to train from scratch
- Hours to days of computation
- Careful hyperparameter tuning

With pre-trained models:
- **100–1,000 labeled cells** often sufficient for fine-tuning
- **Minutes to hours** of computation
- **Default hyperparameters** usually work well

### Learning Rate Selection

When fine-tuning, use a much smaller learning rate than pre-training:
- Pre-training: learning rate ~1e-3
- Fine-tuning: learning rate ~1e-5 to 1e-6

Why? The pre-trained model has already learned good representations. You want to gently adjust these representations for your task, not overwrite them completely.

**Layer freezing**: You can freeze earlier layers and only train later layers. This prevents overfitting when you have limited labeled data. Early layers learn general features (basic gene co-expression patterns), while later layers learn task-specific features (which patterns indicate specific cell types).

---

## 16.7 Embeddings: Capturing Cellular Identity

### What Are Cell Embeddings?

After processing a cell through the model, we get a high-dimensional vector (e.g., 512 dimensions) that represents that cell's transcriptional state. This is the **cell embedding**.

Cells with similar embeddings should be biologically similar:
- Two CD8+ T cells → similar embeddings
- A CD8+ T cell and a hepatocyte → very different embeddings

### Visualizing Embeddings

We can use dimensionality reduction (UMAP, t-SNE) to visualize these embeddings in 2D:

```
High-dim embedding (512-D) → UMAP → 2D plot
```

**Key insight**: Foundation model embeddings often produce cleaner, more interpretable UMAP plots than traditional methods (PCA on raw counts) because they've learned to emphasize biologically relevant variation while ignoring technical noise.

> **Biological analogy:** Gene regulatory network inference from embeddings is like reconstructing who gave instructions to whom in a cell — if gene A's expression always predicts gene B's across millions of cells, A might be upstream of B in the regulatory network. Foundation models make this inference much more reliable by drawing on patterns from far more cells than any single experiment.

### Gene Embeddings

Beyond cells, these models also create **gene embeddings**—vectors representing each gene's functional properties:

- Genes with similar embeddings have similar functions
- Co-regulated genes cluster together
- Transcription factors form a distinct cluster

**Application**: Finding genes with unknown function
1. Get embedding for mystery gene
2. Find nearest neighbors in embedding space
3. Infer function from neighbors

For example, if an uncharacterized gene's embedding is surrounded by cell cycle genes, it likely plays a role in cell division.

### Cross-Species Embeddings

Some foundation models are trained on data from multiple species (human, mouse, zebrafish). This creates **cross-species embeddings** where:

- Human heart cells and mouse heart cells map to similar locations
- Orthologous genes (evolutionary counterparts) have similar embeddings
- We can transfer knowledge from well-studied model organisms to human

This is powerful for studying human disorders where we can't do experiments directly.

---

## Case Study: Identifying Disease-Associated Cell States in Type 2 Diabetes

**Background**: Type 2 diabetes involves dysfunction of pancreatic beta cells, but the molecular mechanisms remain unclear. Different patients show different patterns of beta cell stress and failure.

**The Challenge**: Researchers collected pancreatic islet samples from 50 patients with type 2 diabetes and 20 unaffected controls. After scRNA-seq, they had expression data from 200,000 cells. But which specific beta cell states associate with the disorder?

**Foundation Model Approach**:

**Step 1: Transfer learning**
- Used scGPT pre-trained on 33 million cells
- Fine-tuned on 5,000 labeled beta cells from published studies
- Model learned "typical" beta cell expression patterns

**Step 2: Identify altered states**
- Applied model to patient samples
- Model calculates "distance from typical beta cell state"
- Cells with large distances are in altered states

**Step 3: Cluster altered cells**
- Used model embeddings to cluster the altered beta cells
- Found 4 distinct stressed states:
  1. ER stress state (high expression of ATF4, DDIT3)
  2. Inflammatory state (high IL1B, TNFA)
  3. Dedifferentiation state (low INS, high FOS/JUN)
  4. Senescent state (high CDKN2A, CDKN1A)

**Step 4: Patient stratification**
- Different patients showed different proportions of these states
- This explains clinical heterogeneity

**Key Results**:
- Patients with >20% dedifferentiated beta cells had worst glycemic control
- ER stress state correlated with obesity measures
- These cell states were invisible to traditional analysis

**Clinical Implications**:
- Suggests different patients might benefit from different therapeutic approaches
- ER stress → might respond to PERK inhibitors
- Inflammatory state → might respond to anti-inflammatory treatments
- Enables precision medicine based on cellular phenotypes

**Why Foundation Models Helped**:
- Learned "typical" beta cell state from millions of reference cells
- Could quantify deviations from typical in a principled way
- Found biologically interpretable patterns
- Required minimal manual tuning

---

## 16.8 Multi-Modal Foundation Models

### Beyond RNA: Integrating Data Modalities

The newest generation of foundation models extends beyond scRNA-seq to integrate multiple data types:

**scFoundation** (2024): Combines scRNA-seq and scATAC-seq
- RNA tells us what genes are expressed
- ATAC tells us which chromatin regions are accessible
- Together: predict gene regulation mechanisms

**CellPatch** (2024): Incorporates spatial information from spatial transcriptomics

Traditional scRNA-seq loses spatial context—we don't know where cells were in the tissue. Spatial transcriptomics preserves this, but generates different data types.

CellPatch learns:
- How cell types are spatially organized
- Which cells communicate with neighbors
- How spatial position affects gene expression

**EpiAgent** (2024): Predicts cellular differentiation trajectories

During development, cells transition through intermediate states. EpiAgent learns these temporal relationships.

**Example**: Reprogramming fibroblasts to neurons
1. Start with fibroblast gene expression
2. EpiAgent predicts effects of adding transcription factors
3. Suggests optimal factor combinations
4. Tested experimentally: 3× more efficient than random selection

---

## 16.9 Limitations and Challenges

### What Foundation Models Don't Capture

**Causality**: These models learn correlations, not causal relationships. Just because genes X and Y are always co-expressed doesn't mean X causes Y (they might both be caused by Z).

**Rare cell types**: Models are biased toward common cell types in training data. A cell type that appears in only 1% of training samples will be poorly represented.

**Dynamic processes**: Current models see static snapshots. They struggle with rapidly changing processes (immune responses, cell cycle) where timing matters.

**Cellular interactions**: Single-cell models analyze isolated cells. They miss intercellular signaling, physical contacts, and tissue-level organization (though spatial models are addressing this).

### Technical Challenges

**Computational requirements**:
- Pre-training requires 100+ GPUs for weeks
- Fine-tuning needs 1–8 GPUs for hours
- Inference (applying trained models) can run on CPUs but is slow

**Data quality**:
- "Garbage in, garbage out" applies
- If training data has annotation errors, model learns wrong patterns
- Batch effects must be carefully handled

### Interpretability Challenges

Foundation models are complex. Understanding why they make specific predictions remains difficult:

- Attention weights show what genes the model focuses on, but not why
- Embeddings capture patterns, but may not align with our biological categories
- Models might rely on spurious correlations (artifacts) rather than true biology

**Best practice**: Always validate model predictions experimentally when possible. Use the model to generate hypotheses, not as final proof.

---

## 16.10 Practical Guidelines for Using Foundation Models

### When to Use Pre-Trained Models

**Good use cases**:
- Standard scRNA-seq analysis tasks
- Limited labeled training data available
- Need fast iteration/experimentation
- Working with well-studied tissues/cell types

**Problematic use cases**:
- Highly specialized organisms not in training data
- Novel experimental protocols (spatial, temporal)
- When you need perfect accuracy for clinical decisions
- When interpretability is more important than performance

### Workflow Recommendations

**Start simple**:
1. Try a pre-trained model with zero-shot learning
2. If inadequate, fine-tune with minimal labeled data
3. Only train from scratch if fine-tuning fails

**Validate predictions**:
- Compare to known marker genes
- Check biological plausibility
- Validate key findings experimentally
- Use multiple models and compare results

### Ethical Considerations

**Training data bias**:
- Most models trained primarily on data from individuals of European ancestry
- May not generalize to other populations
- Critical for clinical applications

**Clinical translation**:
- Foundation models are research tools, not clinical diagnostic tools (yet)
- FDA approval required before clinical use
- Must validate on diverse populations

---

## Summary

### Key Takeaways

- **Foundation models** learn general patterns from massive datasets and can be adapted to specific tasks through fine-tuning, requiring far less labeled data than training from scratch

- **scBERT** pioneered BERT-style masked gene prediction for single cells, treating genes as tokens and cells as sequences, achieving strong performance on cell type annotation

- **Geneformer** introduced rank-value encoding and incorporated biological knowledge from Gene Ontology, enabling effective zero-shot learning for new cell types

- **scGPT** added generative capabilities, allowing prediction of cellular responses to perturbations and simulation of drug treatments or genetic modifications

- **Multi-modal models** (scFoundation, CellPatch, EpiAgent) integrate RNA-seq with chromatin accessibility, spatial position, or temporal dynamics to capture additional biological complexity

- **Transfer learning** dramatically reduces data requirements: fine-tuning on 100–1,000 labeled cells often achieves performance that would require 10,000+ cells when training from scratch

- **Cell and gene embeddings** provide interpretable, low-dimensional representations that capture biological similarity and can transfer knowledge across species

- **Limitations** include lack of causal understanding, bias toward common cell types, computational requirements, and interpretability challenges that require careful validation

---

<details>
<summary><strong>📖 Key Terms</strong></summary>

| Term | Definition |
|------|-----------|
| **Attention mechanism** | A neural network component that learns which parts of the input are most relevant for making predictions, allowing the model to focus on important gene-gene relationships |
| **Batch effect** | Technical variation in gene expression measurements arising from different experimental conditions, sequencing runs, or laboratories rather than true biological differences |
| **Cell embedding** | A high-dimensional vector representation of a cell's transcriptional state learned by a neural network, where similar cells have similar embeddings |
| **Conditional generation** | The ability to generate synthetic data with specific properties by conditioning the generative model on desired characteristics (e.g., cell type, treatment condition) |
| **Fine-tuning** | Adapting a pre-trained model to a specific task by training on a small amount of task-specific data with a low learning rate |
| **Foundation model** | A large AI model trained on broad data that can be adapted to many downstream tasks, typically through fine-tuning or few-shot learning |
| **Gene embedding** | A vector representation of a gene that captures its functional properties and regulatory relationships based on co-expression patterns across many cells |
| **Generative model** | A model that learns the probability distribution of data and can generate new synthetic samples, rather than just classifying existing samples |
| **In-context learning** | The ability to perform a task by seeing a few examples without any parameter updates or fine-tuning, analogous to few-shot learning in language models |
| **Masked gene prediction** | A pre-training objective where random genes are hidden and the model learns to predict them from surrounding context, forcing it to learn gene-gene relationships |
| **Multi-modal learning** | Training models on multiple data types simultaneously (e.g., RNA expression and chromatin accessibility) to learn relationships between modalities |
| **Perturbation prediction** | Using trained models to forecast how cells will respond to genetic or chemical interventions without running actual experiments |
| **Rank-value encoding** | Representing gene expression by ranking genes within each cell rather than using absolute expression values, making models more robust to technical variation |
| **Transfer learning** | Training a model on one dataset or task, then applying or adapting it to a different but related task with minimal additional training |
| **Zero-shot learning** | Applying a model to tasks or data types it has never been explicitly trained on, relying on general patterns learned during pre-training |

</details>

---

## Conceptual Questions

1. **Explain why masked gene prediction is an effective pre-training objective for single-cell data.** What biological knowledge does a model gain by learning to predict masked genes from unmasked ones?

2. **Compare rank-value encoding (Geneformer) with absolute expression values (scBERT).** In what scenarios would each approach be more appropriate? What biological assumptions does each make?

3. **A foundation model trained primarily on immune cells is applied to analyze neurons.** What challenges might arise? How would you assess whether the model is making valid predictions?

4. **Consider a model that achieves 98% accuracy on cell type classification but uses attention patterns that don't match known biology.** Is this model trustworthy? What additional validation would you require before using it for hypothesis generation?

5. **How do foundation models handle the scale problem in single-cell biology?** Explain why training one large model on millions of cells is more effective than training separate models for each experiment.

6. **Describe the trade-off between model size and computational efficiency.** When might you choose a smaller, faster model over a larger, more accurate one?

7. **Explain how cell embeddings enable cross-species comparison.** What does it mean for a human T cell and a mouse T cell to have similar embeddings?

8. **Why is fine-tuning with a small learning rate important?** What would happen if you used the same learning rate for fine-tuning as for pre-training?

---

## Further Reading

### Foundational Papers

**scBERT**
- Yang, F., Wang, W., Wang, F., et al. (2022). "scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data." *Nature Machine Intelligence*, 4, 852–866.

**Geneformer**
- Theodoris, C.V., Xiao, L., Chopra, A., et al. (2023). "Transfer learning enables predictions in network biology." *Nature*, 616, 616–624.

**scGPT**
- Cui, H., Wang, C., Maan, H., et al. (2024). "scGPT: toward building a foundation model for single-cell multi-omics using generative AI." *Nature Methods*, 21, 1470–1480.

### Recent Reviews

- Benegas, G., Batra, S.S., & Song, Y.S. (2024). "DNA language models are powerful predictors of genome-wide variant effects." *PNAS*, 121(15), e2311219121.
- Lähnemann, D., Köster, J., Szczurek, E., et al. (2020). "Eleven grand challenges in single-cell data science." *Genome Biology*, 21, 31.

### Online Resources

- CELLxGENE: https://cellxgene.cziscience.com/ (100M+ cells, standardized annotations)
- Human Cell Atlas Data Portal: https://data.humancellatlas.org/
- Geneformer repository: https://huggingface.co/ctheodoris/Geneformer

---

## What's Next

In **Chapter 17: Toward Whole-Cell Modeling**, we'll explore how foundation models are being integrated into comprehensive models of entire cells. We'll see how:

- Multiple data modalities (RNA, chromatin, protein, metabolism) can be integrated into unified cell models
- Dynamic models capture temporal evolution of cellular states
- Spatial models represent tissue architecture and cell-cell interactions
- The *E. coli* whole-cell model demonstrates feasibility of comprehensive modeling
- The Human Cell Atlas aims to create a reference map of all human cell types

Before moving on, make sure you can:

- [ ] Explain the key differences between scBERT, Geneformer, and scGPT
- [ ] Describe when to use pre-trained models vs training from scratch
- [ ] Interpret cell and gene embeddings from foundation models
- [ ] Critically evaluate model predictions and identify when experimental validation is needed
