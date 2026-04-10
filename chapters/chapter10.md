# Chapter 10: Transformer-Based Models for Genomics

**[Interactive: Chapter 10](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter10.html)**

Something puzzling is happening in the β-globin locus. A noncoding variant sits 500,000 base pairs from the *GATA1* gene — half a million bases of seemingly irrelevant DNA in between. Yet somehow, this distant variant causes beta-thalassemia. No local gene is disrupted. No protein is directly altered. The variant is whispering instructions across an enormous genomic distance, and the gene on the other end is listening.

How? The convolutional networks we've studied so far can't answer that question. Their receptive field — the genomic window they can "see" at once — maxes out at roughly 20,000 base pairs, even with many stacked layers. This variant is 25 times beyond that horizon. CNN-based tools like DeepSEA faithfully report the local chromatin state at the variant site, but they're blind to what lies downstream. Asking them about long-range regulation is like asking someone to describe a conversation they can only hear every 25th word of.

Across a dataset of 127 chromatin loops identified by Hi-C in erythroid cells, 83% involve enhancer-promoter pairs separated by more than 100,000 base pairs. Some stretch beyond 1 million bases. These long-range interactions are not edge cases — they are fundamental to how gene regulation works, especially in development and tissue-specific contexts. A model that cannot see across these distances will always be missing half the story.

The answer involves an architectural innovation borrowed from machine translation: a mechanism called *attention* that lets a model look at every position in a sequence simultaneously and learn which distant positions talk to each other. This is exactly the problem transformers were designed to solve — not in genomics initially, but in natural language processing. Welcome to transformers.

---

## The Biological Challenge: Beyond Local Context

### Why Long-Range Interactions Matter

The genome isn't a simple linear instruction manual where each gene operates independently. Instead, it's a complex three-dimensional structure where regulatory elements communicate across vast genomic distances:

**Chromatin Looping in Gene Regulation:**
- Enhancers routinely regulate genes 100,000-1,000,000 bases away
- ENCODE data shows the median enhancer-promoter distance is ~120 kb
- Some regulatory interactions span megabases (multiple genes apart)
- Cell-type-specific gene expression depends on these long-range interactions

Think of it this way: a transcription factor doesn't just check its immediate neighborhood—it scans the entire accessible genome to find its binding partners, wherever they are. Enhancers can "reach across" vast genomic distances to activate a promoter, much like two people can have a conversation across a large room.

**The Splicing Challenge:**
- Human genes average 27,000 bases, but some exceed 2 million bases
- The *dystrophin* gene (DMD) spans 2.2 million bases with 79 exons
- Splicing decisions depend on regulatory elements scattered throughout introns
- Cryptic splice sites can be activated by variants thousands of bases away
- More than 60% of disorder-causing variants in some genes affect splicing

---

## Learning Objectives

By the end of this chapter, you will be able to:

- [ ] Explain why CNNs struggle with long-range genomic interactions and what limits their receptive field
- [ ] Describe how the attention mechanism works and why it enables modeling arbitrary distances
- [ ] Understand the transformer architecture and its key components (self-attention, positional encoding, feedforward layers)
- [ ] Compare transformer-based genomic models (Enformer, Sei) with CNN predecessors in terms of architecture and capabilities
- [ ] Explain how SpliceAI uses dilated convolutions to achieve long-range predictions for splicing
- [ ] Interpret attention weights to understand which genomic positions influence predictions
- [ ] Apply transformer-based models to predict variant effects on gene expression and splicing
- [ ] Recognize when long-range sequence context matters for a biological question

---

## 10.1 The Limits of Convolutional Neural Networks

Before we introduce transformers, let's understand exactly why CNNs face challenges with long-range interactions—and what that means biologically.

### 10.1.1 Receptive Fields in CNNs

Recall from Chapter 9 that CNNs use filters (kernels) that slide across sequences, detecting local patterns. Each convolutional layer has a **receptive field**—the span of input sequence that can influence a single output position.

**How receptive fields grow:**
- Layer 1 with kernel size 3: receptive field = 3 bp
- Layer 2 (stacking on Layer 1): receptive field = 5 bp
- Layer 3: receptive field = 7 bp
- Each additional layer adds (kernel_size - 1) to the receptive field

To reach a receptive field of 100,000 bp with kernel size 3:
- You'd need ~50,000 convolutional layers
- Computationally infeasible (gradient vanishing, memory explosion)

**DeepSEA's approach:**
- Input: 1,000 bp
- Receptive field: entire input (through pooling and multiple layers)
- Can't see beyond its 1,000 bp window

**Basenji's approach (Chapter 9):**
- Input: 131,072 bp (~131 kb)
- Multiple convolutional and pooling layers
- Effective receptive field: ~20,000-40,000 bp
- Better, but still limited for megabase-scale interactions

### 10.1.2 Biological Consequences of Limited Receptive Fields

**Example 1: Enhancer-Promoter Interactions**

The *β-globin* locus control region (LCR) contains enhancers 50-60 kb upstream of the *HBB* gene. Variants in the LCR cause β-thalassemia by reducing *HBB* expression.

- **CNN with 20 kb receptive field**: If analyzing the promoter region, it cannot "see" LCR enhancers
- **Consequence**: Misses critical regulatory variants
- **Experimental validation**: 15-20% of disorder-causing variants affect long-range regulation

**Example 2: Splicing Across Large Genes**

The *DMD* gene (dystrophin) spans 2.2 million bases:
- 79 exons scattered across 2.2 Mb
- Intronic variants can affect splicing of distant exons
- Splicing enhancers and silencers located throughout introns

A variant in intron 20 might affect exon 25 inclusion 50 kb away, but standard CNNs can't integrate this information.

### 10.1.3 Attempts to Extend CNN Receptive Fields

Researchers tried several approaches:

**1. Dilated Convolutions (Atrous Convolutions)**
- Introduce gaps in the filter to cover more distance
- Can exponentially increase receptive field: 3 → 9 → 27 → 81 bp
- Used successfully in SpliceAI (discussed in Section 10.5)
- Trade-off: May miss intermediate positions

**2. Extremely Deep Networks**
- Stack many convolutional layers
- Problems: vanishing gradients, overfitting, computational cost

**3. Larger Input Windows**
- Basenji uses 131 kb inputs
- Memory scales with input length
- Still can't handle megabase-scale interactions

**None of these fully solve the long-range interaction problem.** We need a fundamentally different architecture—one that can relate any position to any other position regardless of distance.

Enter transformers.

---

## 10.2 The Transformer Revolution: From Language to DNA

### 10.2.1 Transformers in Natural Language Processing

Transformers were introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. for machine translation. The key insight: replace sequential processing with parallel processing based on attention mechanisms.

**Why transformers work for language:**
- Words far apart in a sentence can be deeply related
  - "The cat, which I saw at the shelter last week, was adorable."
  - "cat" and "was" are 10 words apart but directly related (subject-verb)
- Traditional RNNs process sequentially, struggling with long-range dependencies
- Transformers use **self-attention** to relate every word to every other word directly

**Biological analogy:**
- Genomic regulatory elements work like words in a sentence
- An enhancer 500 kb away is like a word 10 positions back
- Both require understanding relationships across distance
- Both benefit from attention mechanisms

### 10.2.2 The Self-Attention Mechanism

Self-attention is the core innovation. Here's how it works:

**The Basic Idea:**
For each position in a sequence, attention computes how much to "attend to" every other position. Positions that are relevant get high attention scores; irrelevant positions get low scores.

**Three Components (the QKV framework):**

For each sequence position, we create three vectors:
1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What do I represent?"
3. **Value (V)**: "What information do I carry?"

**How it works step-by-step:**

Imagine analyzing a genomic sequence with three positions (simplified):

```
Position 1: TATA box (promoter element)
Position 2: random intronic sequence  
Position 3: GATA motif (enhancer element)
```

**Step 1: Create Q, K, V vectors**
Each position gets Query, Key, and Value vectors (learned during training):
- Position 1 (TATA): Q₁, K₁, V₁
- Position 2 (random): Q₂, K₂, V₂  
- Position 3 (GATA): Q₃, K₃, V₃

**Step 2: Compute attention scores**
For Position 1, we want to know: "Which other positions are relevant to me?"

Calculate: Q₁ · K₁, Q₁ · K₂, Q₁ · K₃ (dot products)

The dot product measures similarity—high values mean "these positions should interact."

**Step 3: Normalize with softmax**
Convert scores to probabilities that sum to 1:
- Attention(1→1) = 0.2 (some self-attention)
- Attention(1→2) = 0.1 (low—random sequence not relevant)
- Attention(1→3) = 0.7 (high—enhancer-promoter interaction!)

**Step 4: Weighted sum of values**
Output for Position 1 = 0.2 × V₁ + 0.1 × V₂ + 0.7 × V₃

Position 1's representation now incorporates strong information from Position 3 (the enhancer), weak information from Position 2.

**The key insight:** This happens in parallel for ALL positions, and they can all attend to each other regardless of distance.

> **[선택: 수식으로 보면]**
>
> **Attention function:**
>
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
>
> Where:
> - $Q$: Query matrix (sequence_length × d_k)
> - $K$: Key matrix (sequence_length × d_k)  
> - $V$: Value matrix (sequence_length × d_v)
> - $d_k$: dimension of key vectors (scaling factor prevents tiny gradients)
> - $QK^T$: produces attention scores for all position pairs
> - softmax: normalizes to probabilities
> - Result: weighted sum of values based on attention
>
> **Multi-head attention:**
>
> Instead of one attention mechanism, use multiple "heads" in parallel:
> $$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
> where each head$_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
>
> Different heads can specialize: Head 1 might learn promoter-enhancer interactions; Head 2 might learn exon-exon interactions; Head 3 might learn CTCF-CTCF (chromatin loop) interactions.
>
> **Positional Encoding:**
> Transformers process all positions in parallel, losing sequence order information. To preserve order, positional encodings are added to input embeddings:
> $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

---

## 10.3 Enformer: Transformers for Gene Expression Prediction

### 10.3.1 From Basenji to Enformer

**Basenji (Chapter 9 recap):**
- CNN-based architecture
- Input: 131,072 bp (~131 kb)
- Predicts gene expression and chromatin marks
- Limitation: ~20-40 kb effective receptive field

**Enformer (Avsec et al., 2021):**
- Transformer-based architecture  
- Input: 196,608 bp (~197 kb)
- **Key innovation**: Attention mechanism for long-range interactions
- **Result**: 50% better correlation with experimental data than Basenji

### 10.3.2 Enformer Architecture

Enformer combines the best of both worlds: CNNs for local patterns, transformers for long-range interactions.

**Architecture overview:**

```
Input: 196,608 bp DNA sequence
   ↓
[Convolutional Stem]
- 7 convolutional blocks
- Reduces sequence length (downsampling)
- Extracts local features
- Output: 1,536 positions × 1,536 features
   ↓
[Transformer Layers]  
- 11 transformer blocks
- Multi-head self-attention
- Each position can attend to all others
- Learns long-range dependencies
   ↓
[Prediction Heads]
- Predicts 5,313 experimental tracks:
  * CAGE (gene expression)
  * DNase-seq (accessibility)
  * H3K4me3, H3K27ac, etc. (histone marks)
- Outputs: bin-level predictions across sequence
```

**Why this hybrid approach?**
- **CNNs excel at**: Local motifs (TATA boxes, TF binding sites)
- **Transformers excel at**: Long-range interactions (enhancer-promoter)
- **Together**: Best of both—recognize motifs AND their long-distance relationships

### 10.3.3 Enformer for Variant Effect Prediction

**How to predict variant effects:**

1. **Reference prediction:** Input 197 kb sequence centered on gene/variant → get predictions for all 5,313 tracks

2. **Variant prediction:** Input same sequence with variant substituted → get predictions for all tracks

3. **Calculate difference:**
   - Δ expression = prediction(variant) - prediction(reference)
   - Positive Δ: variant increases expression
   - Negative Δ: variant decreases expression

**Example: Enhancer variant analysis**

Variant rs1234567 is 85 kb upstream of *FOXP2* gene. Enformer predicts the variant reduces *FOXP2* expression by 2.3-fold in brain tissue. This long-range effect would be invisible to DeepSEA or standard Basenji, but Enformer's attention mechanism can integrate the enhancer-promoter connection across 85 kb.

### 10.3.4 Enformer Performance

Performance metrics (Avsec et al., 2021):

Correlation with experimental data:
- Basenji: 0.62 (Pearson correlation)
- Enformer: 0.81 (Pearson correlation)
- **Improvement: 31% better correlation**

By visualizing attention weights, researchers discovered Enformer learned:

1. **Promoter-enhancer grammar:** Strong attention between enhancer marks (H3K27ac) and promoters, matching known chromatin loops from Hi-C

2. **CTCF-mediated loops:** Attention between convergent CTCF sites, matching known chromatin domain boundaries

3. **Splicing regulatory elements:** Attention between exons and distant intronic elements

**This is remarkable:** Enformer learned 3D chromatin organization from linear DNA sequence alone, without any Hi-C data during training.

---

## 10.4 Sei: Sequence-to-Chromatin State Prediction

### 10.4.1 The Chromatin State Segmentation Challenge

Chromatin states represent distinct functional categories of genomic regions:
- **Active promoter:** H3K4me3+, H3K27ac+, DNase+
- **Strong enhancer:** H3K4me1+, H3K27ac+, DNase+
- **Poised enhancer:** H3K4me1+, H3K27me3+
- **Heterochromatin:** H3K9me3+, repressed
- **Transcribed region:** H3K36me3+

**The Sei approach:**
- Predicts chromatin states directly from sequence
- No experimental data needed for new cell types
- Uses deep learning with attention

### 10.4.2 Sei Architecture

**Sei (Chen et al., 2022)** combines CNN encoding with attention pooling:

```
Input: 4,096 bp DNA sequence
   ↓
[CNN Encoder] — extracts local sequence features
   ↓
[Attention Pooling] — integrates information across sequence
   ↓
[Prediction Heads]
- Predicts 21,907 TF binding profiles
- Predicts chromatin accessibility
- Predicts histone modification patterns
   ↓
[Chromatin State Classification]
- Maps predictions to 40 distinct chromatin states
- Tissue-specific state predictions
```

### 10.4.3 Sei for Noncoding Variant Interpretation

**Sei's approach:**

1. **Sequence class scoring:** Each sequence gets a "class score" for chromatin states

2. **Variant effect as class shift:**
   - Reference: "active enhancer" score = 0.92
   - Variant: "active enhancer" score = 0.23
   - Δ = -0.69: variant disrupts enhancer function

**Example application:**

Variant rs7903146 near *TCF7L2* (type 2 diabetes GWAS hit):

```
Reference sequence:
- Active enhancer (pancreatic islet): 0.88
- Weak enhancer (liver): 0.34
- Quiescent (brain): 0.02

Variant sequence:
- Active enhancer (pancreatic islet): 0.12  ← decreased!
- Weak enhancer (liver): 0.31  ← minimal change
- Quiescent (brain): 0.03  ← no change

Interpretation: Variant disrupts enhancer specifically in 
pancreatic islet cells, where TCF7L2 regulates insulin 
secretion. This matches the tissue specificity of type 2 
diabetes.
```

### 10.4.4 Interpretability Through Attention

One advantage of attention mechanisms: **interpretability**.

For a sequence predicted as "strong enhancer":
- High attention to position 500-510 (AP-1 motif)
- High attention to position 1200-1210 (ETS motif)
- Low attention to other regions
- **Interpretation:** AP-1 and ETS binding drive enhancer activity

---

## 10.5 SpliceAI: Predicting Splicing from Sequence

### 10.5.1 The Splicing Prediction Challenge

**Pre-mRNA splicing:**
- Removes introns, joins exons
- Critical for gene expression
- Highly regulated, tissue-specific
- Errors cause ~15% of genetic disorders

**Splicing signals:**
- **Splice sites:** GT-AG consensus at exon-intron boundaries
- **Exonic splicing enhancers (ESEs):** Promote exon inclusion
- **Intronic splicing silencers (ISSs):** Suppress cryptic sites

**The challenge:** Splicing signals can be >1,000 bp from splice sites. Deep intronic variants can activate cryptic sites. The *DMD* gene spans 2.2 million base pairs, with variants deep in introns capable of causing Duchenne muscular dystrophy.

### 10.5.2 SpliceAI Architecture

**SpliceAI (Jaganathan et al., 2019)** uses dilated convolutions to achieve long-range predictions without a full transformer architecture.

```
Input: Up to 10,000 bp sequence (centered on potential splice site)
   ↓
[Residual Blocks with Dilated Convolutions]
Block 1: dilation = 1 (local context)
Block 2: dilation = 2
Block 3: dilation = 4  
Block 4: dilation = 8
...
Block 8: dilation = 128
   ↓
[Output Predictions at Each Position]
- Acceptor splice site probability
- Donor splice site probability  
- Probability position is in exon
```

**Why this works:**
- Early layers (dilation=1,2): detect local motifs (GT-AG, ESEs)
- Later layers (dilation=16,32): integrate distant regulatory elements
- Residual connections: prevent vanishing gradients in deep network
- Total receptive field: ~10,000 bp

### 10.5.3 SpliceAI Predictions and Scores

**SpliceAI score for variants:**

- 0.0-0.2: Low impact
- 0.2-0.5: Moderate impact  
- 0.5-0.8: High impact
- 0.8-1.0: Very high impact (likely disrupts splicing)

**Types of splicing effects detected:**

1. **Splice site disruption:** Variant at GT or AG → loss of canonical splice site → score ~0.9-1.0
2. **Cryptic splice site activation:** Variant creates new GT-AG or strengthens weak cryptic site → score 0.5-0.9
3. **Exon skipping:** Variant in ESE weakens exon definition → exon skipped in mature mRNA → score 0.3-0.7
4. **Intron retention:** Variant weakens splice sites → intron retained in mRNA → score 0.3-0.6

### 10.5.4 SpliceAI in Clinical Genetics

**Performance on known splicing variants:**

Jaganathan et al. (2019) tested SpliceAI on:
- 27,733 variants with experimental splicing data
- Area under ROC curve: 0.98
- Sensitivity: 95% at 95% specificity
- **Far better than previous tools (MaxEntScan, etc.)**

**Example: Synonymous variant with splicing impact**

Patient with cystic fibrosis:
- Known *CFTR* variant: c.1584G>A
- Amino acid change: None (synonymous, L→L)
- Previously classified: likely neutral
- SpliceAI score: 0.87 (disrupts ESE)
- Functional studies: confirmed exon 11 skipping
- **Reclassified as having functional impact**

ClinVar now includes SpliceAI scores for all variants, helping prioritize candidates and identify splicing as a mechanism.

---

## 10.6 Attention Mechanisms: What Transformers Learn

### 10.6.1 Visualizing Genomic Attention

One of the most exciting aspects of transformer models is that we can visualize what they learn by examining attention weights.

**What are attention weights?**
- Quantify how much each position "attends to" every other position
- Ranges from 0 (no attention) to 1 (maximum attention)
- Can be visualized as heatmaps

**Example: Enformer attention for β-globin locus**

```
Analyzing HBB gene promoter region:
     Promoter  LCR      Intergenic  Downstream
Pos:  [0]    [50kb]     [75kb]      [100kb]

Attention weights from promoter position:
- Self (promoter): 0.15
- LCR (50 kb away): 0.78  ← Strong attention!
- Intergenic: 0.02
- Downstream: 0.05

Interpretation: Model learned enhancer-promoter 
interaction between LCR and HBB promoter.
```

**Comparing with Hi-C data:**
- Enformer attention correlates with chromatin loops (Pearson r = 0.73)
- High attention = predicted physical interaction
- **No Hi-C data used in training—emerged from sequence!**

### 10.6.2 Attention Patterns Reveal Regulatory Grammar

**Discovered patterns:**

1. **Promoter-Enhancer Syntax:** Enhancers with H3K27ac marks attend to nearest promoters; attention blocked by CTCF insulators, matching topologically associating domains (TADs)

2. **Splice Site Recognition:** Attention between donor and acceptor sites; strong attention within exons (exon definition); attention to regulatory elements (ESEs, ISSs)

3. **Motif Co-occurrence:** AP-1 and ETS motifs attend to each other, suggesting cooperative TF binding—this matches known TF partnerships

### 10.6.3 Limitations of Attention Interpretability

**Attention is not causation:**
- High attention doesn't prove a functional relationship
- May reflect correlation, not mechanism
- Need experimental validation

**Best practice:**
- Use attention as hypothesis-generating tool
- Combine with other evidence (conservation, functional data)
- Validate computationally-driven hypotheses experimentally

---

## Case Study 1: Enformer Predicts Enhancer-Promoter Interactions

### Background

Avsec et al. (2021) tested whether Enformer could predict enhancer-promoter interactions without explicit Hi-C training data.

**Experimental setup:**
- Selected 1,000 known enhancer-promoter pairs from Hi-C
- Pairs span 10 kb to 500 kb distances
- Across multiple cell types (GM12878, K562, HepG2)

### Results

**Correlation with Hi-C:** Pearson r = 0.73 between attention and loop strength. Stronger loops → higher attention weights. Tissue-specific: attention matches cell-type-specific loops.

**Validation with CRISPR:**
- Selected 10 predicted interactions
- Used CRISPRi to silence enhancers
- Measured target gene expression
- 9/10 showed expected expression changes
- **Computational predictions validated experimentally**

### Biological Insights

**Discovered regulatory grammar:**
1. **Orientation matters:** Convergent CTCF sites show high attention, matching known loop extrusion model—learned without explicit CTCF training
2. **Enhancer strength predicts attention:** H3K27ac signal correlates with attention; strong enhancers attend more to promoters
3. **Promoter competition:** Enhancers between two promoters split attention, matching models of enhancer sharing

---

## Case Study 2: SpliceAI Identifies Causative Variant in Undiagnosed Patient

### Clinical Scenario

**Patient presentation:**
- 8-year-old with progressive muscle weakness
- Family history: negative
- Symptoms: difficulty walking, elevated creatine kinase
- Suspected: muscular dystrophy

**Genetic testing:**
- Duchenne/Becker muscular dystrophy panel: negative
- Whole-exome sequencing: negative (no coding variants in known genes)
- Next step: genome sequencing

### Genome Analysis

**Initial findings:**
- 4.3 million variants total
- After filtering: 127 rare variants in genes related to muscle function
- Manual review: no obvious candidates in coding regions

**SpliceAI analysis:**
- Ran SpliceAI on all 127 variants
- One variant flagged: c.663+784C>T in *DYSF* (dysferlin gene)
- Location: 784 bp into intron 7 (deep intronic)
- SpliceAI score: 0.91 (very high)

**Prediction details:**
```
Reference sequence:
- No cryptic splice sites predicted
- Normal exon 7 - exon 8 splicing

Variant sequence:  
- Creates cryptic donor site (GT) at +784
- Predicts pseudoexon inclusion
- 47 bp insertion in mRNA
- Frameshift → premature stop codon
```

### Experimental Validation

**RT-PCR from patient muscle biopsy:**
- Results: aberrant transcript with 47 bp insertion confirmed
- Sequencing: exactly as predicted

**Protein analysis:**
- Western blot: reduced dysferlin protein
- Diagnosis: **dysferlinopathy** (limb-girdle muscular dystrophy type 2B)

### Impact

**For this patient:**
- Ended diagnostic odyssey (4 years of uncertainty)
- Enabled family planning (genetic counseling)
- Informed prognosis and management

**Broader implications:**
- ~5-10% of unsolved cases may harbor splicing variants
- Many clinical labs now run SpliceAI routinely
- Integrated into variant interpretation pipelines

---

## Summary: Key Takeaways

### Architectural Innovations

- **CNNs struggle with long-range interactions** due to limited receptive fields, typically <40 kb even in deep networks
- **Transformers use self-attention** to relate any position to any other position, regardless of distance
- **Hybrid architectures** (CNN + transformer) combine local pattern recognition with long-range integration
- **Dilated convolutions** offer an alternative approach to extend receptive fields without full transformer complexity

### Model Applications

- **Enformer** predicts gene expression and chromatin states from 197 kb sequences, achieving 50% better performance than CNN-based Basenji
- **Sei** classifies chromatin states from 4 kb sequences, enabling noncoding variant interpretation
- **SpliceAI** predicts splicing effects with 98% accuracy using dilated convolutions and 10 kb context
- **Attention weights** provide interpretability, revealing learned regulatory grammar

### Biological Insights

- **Long-range regulation is pervasive**: enhancers regulate genes 50-500 kb away, critical for tissue-specific expression
- **Splicing depends on distant elements**: regulatory sequences >1 kb from splice sites commonly affect splicing
- **Models learn regulatory grammar**: attention patterns match chromatin loops, CTCF boundaries, and TF cooperativity

### Limitations to Remember

- **Computational cost**: transformers require more memory and compute than CNNs
- **Context limits**: even transformers have maximum sequence lengths (197 kb for Enformer)
- **Validation essential**: computational predictions must be experimentally validated for clinical use

---

<details>
<summary><strong>📖 Key Terms</strong></summary>

| Term | Definition |
|------|-----------|
| **Attention mechanism** | Method for computing relevance of different positions in a sequence to each other, enabling long-range dependency modeling regardless of distance. |
| **Attention weight** | Numerical score (0 to 1) quantifying how much one position "attends to" another position; high values indicate strong predicted interactions. |
| **Chromatin loop** | Physical interaction between distant genomic regions mediated by protein complexes; often brings enhancers and promoters into proximity. |
| **Cryptic splice site** | Sequence resembling a splice site (GT-AG) that is normally not used but can be activated by variants or regulatory changes. |
| **Dilated convolution** | Convolutional operation with gaps between kernel positions, enabling larger receptive fields without increasing parameters. |
| **Enformer** | Transformer-based model predicting gene expression and chromatin features from 197 kb DNA sequences; achieves 50% improvement over CNN-based methods. |
| **Exonic splicing enhancer (ESE)** | Sequence element in exons that promotes exon inclusion through binding of SR proteins. |
| **Intronic splicing silencer (ISS)** | Sequence element in introns that suppresses splice site recognition, preventing cryptic splicing. |
| **Multi-head attention** | Parallel attention mechanisms (heads) that can learn different types of relationships; outputs are combined for final representation. |
| **Positional encoding** | Mathematical representation of sequence position added to input embeddings, preserving order information in transformers. |
| **Query-Key-Value (QKV)** | Framework for attention computation where queries search for relevant keys to retrieve corresponding values. |
| **Receptive field** | Span of input sequence that can influence a single output position; larger receptive fields capture longer-range dependencies. |
| **Sei** | Sequence-based chromatin state prediction model using attention mechanisms; predicts 40 chromatin states from 4 kb sequences. |
| **Self-attention** | Attention mechanism where a sequence attends to itself, computing relationships between all position pairs within the same sequence. |
| **SpliceAI** | Deep learning model predicting splicing effects using dilated convolutions; achieves 10 kb receptive field for splice site prediction. |
| **Transformer** | Neural network architecture based on self-attention mechanisms, originally developed for natural language processing, now applied to genomics. |

</details>

---

## Test Your Understanding

1. **Receptive field limitations**: Explain why a CNN with 20 convolutional layers (kernel size 3) still cannot effectively model interactions between an enhancer 100 kb upstream and its target promoter. What specifically limits the flow of information?

2. **Attention vs convolution**: A researcher argues that "attention is just a fancy way of doing convolution across the entire sequence." Explain why this is incorrect and what fundamentally distinguishes the attention mechanism from convolution.

3. **Biological relevance of long-range modeling**: The dystrophin gene (*DMD*) spans 2.2 million bases. Would Enformer (197 kb input) be sufficient to analyze all potential regulatory interactions for this gene? Why or why not? What biological features might be missed?

4. **Attention interpretability**: When Enformer shows high attention between two genomic positions 80 kb apart, what can we confidently conclude, and what remains uncertain? How would you validate that this attention reflects a functional regulatory interaction?

5. **Tissue-specific predictions**: SpliceAI achieves 98% accuracy predicting splicing but doesn't explicitly model tissue types. How can it predict splicing so accurately given that splicing is often tissue-specific? What information in the sequence might enable tissue-agnostic prediction?

6. **Dilated convolutions vs transformers**: SpliceAI uses dilated convolutions while Enformer uses transformers, both achieving long-range predictions. Compare these approaches: What are the computational trade-offs? When might you prefer one over the other?

7. **Variant effect prediction**: A noncoding variant 150 kb upstream of a gene shows a large predicted effect in Enformer but is absent from GWAS studies of any trait. What might explain this discrepancy? What additional evidence would you seek?

8. **Multi-head attention specialization**: Enformer uses 8 attention heads per layer. What biological advantage might there be to having multiple parallel attention mechanisms rather than a single, larger attention mechanism?
