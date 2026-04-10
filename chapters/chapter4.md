# Chapter 4: Neural Network Architectures

**[Interactive: Chapter 4](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter4.html)**

---

## Opening Vignette: Three Datasets, Three Challenges

Dr. Sarah Chen faces a computational puzzle. Her lab generated three datasets to understand how gene regulation works during T cell differentiation:

**Dataset 1: ChIP-seq peaks (125,000 genomic regions)**  
She needs to identify transcription factor binding motifs within these sequences. Each region is 200 base pairs long. The motifs are short (8-15 bp), position-specific patterns buried within the larger sequence. Traditional position weight matrices miss subtle patterns.

**Dataset 2: RNA-seq time course (20,000 genes × 8 time points)**  
Gene expression measured every 6 hours during T cell activation. She wants to predict expression at the next time point, but expression at hour 24 depends on what happened at hours 0, 6, 12, and 18—not just the previous measurement.

**Dataset 3: Hi-C chromatin interactions (50,000 genomic loci)**  
Which distant enhancers physically contact which promoters? An enhancer at position 1,000,000 might regulate a gene at position 1,500,000—500 kilobases away. Local information isn't enough; the model needs to "see" long-range dependencies.

She trained three different neural networks using the techniques from Chapter 3. All failed:
- The fully-connected network for sequences had 200 × 4 = 800 input features per position, creating millions of parameters. It found no patterns.
- The network for time series treated each time point independently, ignoring temporal order.  
- The network for chromatin interactions could only look at nearby regions, missing distant regulatory elements.

**Each dataset has different structure. Why should one architecture work for all of them?**

**Connection to Chapter 2 & 3:** Remember from Chapter 2 that biology is fundamentally probabilistic—gene expression follows distributions, not fixed values. In Chapter 3, we learned how basic neural networks learn through Bayesian updating. But now we face a new challenge: **the structure of biological data matters**. We need architectures that respect this structure.

---

## The Biological Challenge

The biological world generates diverse data types, each with unique structural properties:

**Spatial patterns in sequences:** Transcription factor binding sites, splice sites, and regulatory elements are position-specific patterns. A motif at position 50-58 carries information, but the same motif shifted to position 100-108 means something different. Fully-connected networks treat every position as independent features—destroying spatial relationships.

**Sequential dependencies in time series:** Gene expression, cell state trajectories, and developmental processes unfold over time. Expression at time *t* depends on history: what happened at times *t*-1, *t*-2, *t*-3. Treating each time point independently loses this temporal structure.

**Long-range dependencies:** Enhancers regulate genes megabases away. Alternative splicing depends on splicing signals hundreds of nucleotides apart. 3D genome organization brings distant loci together. Standard neural networks have limited "receptive fields"—they can't see these long-range interactions.

**Why not just use fully-connected networks for everything?**

- **Exploding parameters:** A 1000bp sequence with fully-connected layers needs 1000 × 4 × hidden_size weights—millions of parameters even for simple tasks
- **No built-in structure:** They don't "know" that nearby positions in sequences are related, or that time has order, or that long-range dependencies exist
- **Poor generalization:** They memorize specific positions rather than learning reusable patterns

**The solution:** Specialized neural network architectures that match the structure of biological data.

---

## Learning Objectives

By the end of this chapter, you will be able to:

- [ ] Explain why different biological data types require different neural network architectures
- [ ] Describe how convolutional neural networks (CNNs) detect patterns in sequences and images
- [ ] Understand how recurrent neural networks (RNNs) process sequential data with temporal dependencies
- [ ] Explain the "vanishing gradient problem" and how LSTMs solve it
- [ ] Describe how Transformers use attention mechanisms to capture long-range dependencies
- [ ] Choose the appropriate architecture for different biological problems

---

## 4.1 Convolutional Neural Networks (CNNs): Detecting Local Patterns

> **[생물학적 비유]** CNNs are like scanning a histology slide section by section, detecting cell patterns in small windows—the same pattern detector slides across the entire sample, regardless of position.

### The Core Idea: Reusable Pattern Detectors

Imagine you're looking for transcription factor binding motifs in DNA sequences. A motif like "GATAAG" might appear anywhere in the sequence—position 10, position 150, position 800. You don't want to learn three different "GATAAG detectors" for three different positions.

**CNNs solve this with a simple idea:** Use the same pattern detector (called a filter or kernel) and slide it across the entire sequence, like using a magnifying glass to scan a document.

**Connection to Chapter 3:** Remember how individual neurons in Chapter 3 had weights for each input? CNNs use the same set of weights at every position—this is called **weight sharing**, and it dramatically reduces the number of parameters!

### How Convolution Works

Think of it like this: You have a small window (the filter) that you slide along the DNA sequence, checking for a specific pattern at each position.

**Step 1: Define a filter (kernel)**

A filter is a small set of learnable weights. For DNA sequences, a filter might be 8 bp wide:

```
Filter (length 8):
[w₁, w₂, w₃, w₄, w₅, w₆, w₇, w₈]
```

**Step 2: Slide the filter across the sequence**

Apply the filter to every position:

```
DNA sequence: ATCGATAAGCCGTA...
Position 1:   ATCGATAA → compute score
Position 2:    TCGATAAG → compute score  
Position 3:     CGATAAGC → compute score
...
```

At each position, multiply filter weights by sequence features and sum them up.

**Step 3: Apply activation function**

```
Activated score = ReLU(Score + bias)
```

**Result:** A "feature map" showing where the pattern appears:

```
Feature map: [0.2, 0.0, 0.0, 8.5, 7.3, 0.1, ...]
                              ↑    ↑
                    High activations = pattern detected!
```

### Example: GATA Motif Detection

Let's detect "GATAAG" in a real sequence.

**Input sequence:**
```
Position:  1  2  3  4  5  6  7  8  9  10 11 12
Sequence:  A  T  C  G  A  T  A  A  G  C  C  G
```

**After training, the CNN learns that certain positions should have high scores when the GATA pattern appears.**

At position 5-9 (where "GATAAG" is present), the filter gives a high score (~9.3).  
At other positions without the pattern, the score is low (~0.2).

**The CNN automatically learns which weights detect important patterns by training on labeled data!**

### Key CNN Concepts

#### Multiple filters = Multiple pattern detectors

Real CNNs use dozens or hundreds of filters simultaneously:
- Filter 1 learns "GATA" motif
- Filter 2 learns "TATA" box
- Filter 3 learns "CCAAT" box  
- Filter 4 learns "CpG island" patterns
- ... etc.

Each filter creates one feature map, so 100 filters create 100 feature maps showing different patterns.

#### Pooling: Summarizing information

After convolution, we often use **max pooling** to:
1. Reduce the amount of data (faster computation)
2. Make detection position-invariant (whether motif is at position 50 or 52 doesn't matter much)

```
Before pooling: [0.2, 0.0, 8.5, 7.3, 0.1, 0.3]
Max pooling (window=2): [0.2, 8.5, 0.3]
                         ↑    ↑    ↑
                    Take max from each pair
```

**Biological intuition:** "Is there a GATA motif somewhere in this region?" is often more important than "Is there a GATA motif exactly at position 57?"

#### Stacking layers builds complexity

- **Layer 1 filters:** Detect simple patterns (single motifs like "GATA")
- **Layer 2 filters:** Combine Layer 1 patterns (motif combinations like "GATA + CCAAT")  
- **Layer 3 filters:** Detect higher-level structures (complete regulatory modules)

This is **hierarchical feature learning**—just like how your visual system detects edges → shapes → objects!

### CNN Architecture for Genomics

A typical CNN for sequence analysis:

```
Input: DNA sequence (1000 bp × 4 channels)
    ↓
Conv Layer 1: 128 filters, size 8
    ↓ (128 feature maps)
ReLU + Max Pool (window=2)
    ↓
Conv Layer 2: 64 filters, size 8  
    ↓ (64 feature maps)
ReLU + Max Pool (window=2)
    ↓
Flatten → Fully Connected Layer → Output
```

**Parameter efficiency:**
- Input: 1000 × 4 = 4000 values
- Conv1: Only 8 × 4 × 128 = 4,096 weights (reused at all positions!)
- Compare to fully-connected: Would need 4000 × hidden_size weights

> **[선택: 수식으로 보면]**
> 
> At each position *i*, the convolution output is:
> 
> y(i) = ReLU( Σⱼ w(j) · x(i+j) + b )
> 
> where w is the filter weight vector, x is the input sequence, and b is a bias term. The same w and b are used at every position—this is weight sharing.

### Real Applications: DeepSEA

**What is DeepSEA?** A CNN-based model that predicts chromatin features directly from DNA sequence ([Zhou & Troyanskaya 2015, Nature Methods](https://doi.org/10.1038/nmeth.3547)).

**Task:** Given a 1000bp DNA sequence, predict 919 chromatin features:
- Transcription factor binding
- Histone modifications (H3K4me3, H3K27ac, etc.)
- DNase hypersensitivity
- Across 125 cell types

**Training data:** DeepSEA and similar CNN models were trained on massive public epigenome datasets:

> **Major Genomics Consortia Data**
>
> **ENCODE (Encyclopedia of DNA Elements)**  
> Goal: Catalog all functional elements in the human genome  
> Data: >10,000 experiments across cell types  
> Includes: TF binding, histone marks, chromatin accessibility, RNA expression  
> Why it matters: Provides ground truth labels for training AI models
>
> **Roadmap Epigenomics**  
> Goal: Map epigenomic landscapes across human tissues  
> Data: 111 reference epigenomes  
> Focus: DNA methylation, histone modifications, chromatin states  
> Why it matters: Shows how the same DNA sequence has different functions in different cell types
>
> **FANTOM (Functional Annotation of the Mammalian Genome)**  
> Goal: Identify transcription start sites and enhancers  
> Data: Cap Analysis Gene Expression (CAGE) across hundreds of samples  
> Why it matters: Defines where transcription actually begins genome-wide

**Why it matters:** A single nucleotide change can disrupt a binding site. DeepSEA can predict this effect without doing expensive experiments!

### When to Use CNNs

✅ **Good for:**
- DNA/RNA/protein sequences (motif detection)
- ChIP-seq peak analysis
- Splice site prediction
- Microscopy images (cell classification)
- Any data with local spatial patterns

❌ **Not ideal for:**
- Time series where global order matters (use RNN/LSTM)
- Very long-range dependencies (>1000bp apart, use Transformer)
- Small datasets (CNNs need lots of training data)

---

## 4.2 Recurrent Neural Networks (RNNs): Processing Sequential Data

> **[생물학적 비유]** RNNs are like reading a protein sequence one amino acid at a time, updating your interpretation as you go—each new residue informs your understanding of the whole chain's function.

### The Problem: Time Matters

Imagine predicting gene expression at hour 24 based on measurements at hours 0, 6, 12, 18:

```
Hour:        0    6    12   18   24
Expression:  2.3  4.1  7.8  6.2  ???
```

**The challenge:** Expression at hour 24 doesn't just depend on hour 18—it depends on the entire trajectory. Did expression increase gradually (2.3→4.1→7.8) or spike suddenly? This history matters!

**CNNs can't handle this well because:**
- They look at fixed windows (e.g., hours 12-18)
- They don't "remember" what happened at hour 0
- They process everything at once, not sequentially

**We need a network that processes sequences one step at a time, maintaining memory of what came before.**

### The Core Idea: Hidden State as Memory

An RNN maintains a **hidden state** that gets updated at each time step. Think of it as the network's "memory" or "notes" that it carries forward.

```
h₀ (initial memory: empty)
  ↓
x₁ → [RNN] → h₁ → output₁
       ↓
x₂ → [RNN] → h₂ → output₂
       ↓
x₃ → [RNN] → h₃ → output₃
```

At each step:
1. Take current input (e.g., expression at hour 6)
2. Take previous memory (what happened before)
3. Combine them to create updated memory
4. Optionally produce an output

**The hidden state is like taking notes as you read a story—each new sentence updates your understanding!**

### How RNNs Work (Simplified)

At each time step, the RNN does two things:

1. **Update memory:**
```
new_memory = combine(current_input, previous_memory)
```

2. **Produce output:**
```
output = process(new_memory)
```

**Key point:** The same processing happens at every time step—the network uses the same "rules" throughout the sequence.

### Example: Predicting Gene Expression

Let's predict expression at hour 24:

```
Input sequence: [2.3, 4.1, 7.8, 6.2]  (hours 0, 6, 12, 18)
Target: 5.5  (hour 24)
```

**Processing step by step:**

**Hour 0:** Input: 2.3 → Memory: "I saw expression = 2.3"

**Hour 6:** Input: 4.1 + Previous memory → Memory: "I saw 2.3, then 4.1 (increasing trend!)"

**Hour 12:** Input: 7.8 + Memory → Memory: "Expression is rising: 2.3 → 4.1 → 7.8"

**Hour 18:** Input: 6.2 + Memory → Memory: "Rose to 7.8, then dropped to 6.2 (peak and decline!)"

**Final prediction:** Based on memory "peaked at 7.8, now declining to 6.2" → Prediction: ~5.5

The final memory contains information about the entire trajectory!

> **[선택: 수식으로 보면]**
>
> At each step *t*, the RNN update is:
>
> h(t) = tanh( Wₓ · x(t) + Wₕ · h(t-1) + b )
>
> output(t) = Wₒ · h(t)
>
> where Wₓ, Wₕ, Wₒ are weight matrices shared across all time steps.

**Connection to Chapter 2:** This is like Bayesian updating! Each new observation updates the network's "belief" (hidden state) about what's happening.

### The Vanishing Gradient Problem

RNNs have a serious limitation: **they forget long-term dependencies.**

**Why?** During training, the network needs to learn from examples that happened many steps ago. But as gradients flow backward through time, they get weaker and weaker—like a whisper that fades as it travels through a long corridor.

**In practice:** Basic RNNs can only remember ~10 time steps back. For biology, this is problematic:
- Splice sites can be 100+ nucleotides apart
- Regulatory interactions span kilobases
- Cell differentiation involves many sequential decisions

**Imagine trying to predict the ending of a book but only remembering the last 2 pages—that's the vanishing gradient problem!**

**Solution:** Long Short-Term Memory (LSTM) networks.

---

## 4.3 Long Short-Term Memory (LSTM): Solving the Memory Problem

> **[생물학적 비유]** LSTMs are like immunological memory—they can retain information about early antigens even after many cell divisions, selectively keeping important signals while discarding noise.

### The Core Idea: Selective Memory with Gates

LSTMs solve the forgetting problem with a clever mechanism: **gates** that control information flow.

Think of memory like a notepad where you can:
- **Erase** old information (forget gate)
- **Write** new information (input gate)
- **Read** selected information (output gate)

**Unlike basic RNNs that gradually forget everything, LSTMs actively choose what to remember and what to forget!**

### How LSTMs Work (Intuitive Explanation)

At each time step, an LSTM asks three questions:

**1. What should I forget?** (Forget gate)
- "The gene was highly expressed 5 hours ago, but that doesn't matter anymore"
- Controlled by a forget gate (values 0 to 1)
  - 0 = completely forget
  - 1 = completely keep

**2. What should I remember?** (Input gate)
- "A transcription factor just bound—this is important!"
- Controlled by an input gate (values 0 to 1)
  - 0 = ignore this new information
  - 1 = store this information

**3. What should I output?** (Output gate)
- "Based on everything I know, what's relevant right now?"
- Controlled by an output gate (values 0 to 1)
  - 0 = don't output this memory
  - 1 = output this memory

### LSTM Architecture (Simplified)

An LSTM has two types of memory:

```
Cell state (C): Long-term memory storage
    ↓
Hidden state (h): What's currently active/relevant
```

**The cell state is like a conveyor belt that carries information forward, with gates deciding what gets added or removed along the way.**

> **[선택: 수식으로 보면]**
>
> The three gates are computed as:
>
> - Forget gate: f(t) = σ( Wf · [h(t-1), x(t)] + bf )
> - Input gate: i(t) = σ( Wi · [h(t-1), x(t)] + bi )
> - Output gate: o(t) = σ( Wo · [h(t-1), x(t)] + bo )
>
> Cell state update: C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ tanh( Wc · [h(t-1), x(t)] + bc )
>
> where σ is the sigmoid function (outputs 0–1) and ⊙ is element-wise multiplication.

### Example: Remembering Splice Sites

Imagine an LSTM processing a gene sequence to predict splice sites:

**Position 1-50: Exon sequence**
```
Memory: "I'm in an exon" (stored in cell state)
Output: "Not a splice site"
```

**Position 100: Donor site (GT)**
```
Forget gate: "Keep exon memory" (forget = 0.9, keep most of it)
Input gate: "Remember this GT!" (input = 1.0, store strongly)
Updated memory: "Was in exon, now saw donor site GT"
Output: "This is a donor splice site!"
```

**Position 101-500: Intron sequence**
```
Forget gate: "GT is still relevant, but exon info can fade" 
Cell state: Maintains "GT donor site" memory across 400 nucleotides
```

**Position 530: Acceptor site (AG)**
```
Cell state still remembers: "Saw GT donor 430 nt ago"
Input gate: "Remember this AG!"
Output: "This AG pairs with the GT I saw earlier—it's an acceptor!"
```

**The key insight:** The cell state carried "GT donor" information across 430 nucleotides—something basic RNNs can't do!

### Why LSTMs Solve Vanishing Gradients

- **Basic RNN:** Gradients multiply at each step → become tiny
- **LSTM:** Gradients can flow directly through the cell state → stay strong
- **Intuitively:** Cell state is a protected "highway" where information travels unchanged

### Real Applications: SpliceAI

SpliceAI (Jaganathan et al., 2019) predicts splice sites using deep networks:

**Task:** Given a gene sequence, predict:
- Splice donor sites (GT)
- Splice acceptor sites (AG)
- Neither

**Challenge:** Splice sites depend on regulatory elements 100-500 nucleotides away.

**Impact:** Predicts how variants affect splicing—a mutation 200bp from an exon can disrupt splicing by affecting a regulatory element.

### Bidirectional LSTMs: Reading Both Directions

For many biological sequences, context matters in both directions:

```
← Look backward
A T C G [?] T A G C  
→ Look forward
```

**Bidirectional LSTM:** Process the sequence twice:
1. Left-to-right (forward direction)
2. Right-to-left (backward direction)
3. Combine both at each position

**Example:** Protein secondary structure prediction needs context from both amino acids before and after each position.

### When to Use RNNs/LSTMs

✅ **Good for:**
- Time series (gene expression over time)
- Sequential data where order matters
- Variable-length sequences
- Predicting next step based on history

❌ **Not ideal for:**
- Very long sequences (>500 steps, use Transformers)
- When you need fast training (RNNs are slow)
- Spatial patterns without temporal order (use CNN)

---

## 4.4 Transformers: Attention is All You Need

> **[생물학적 비유]** The Transformer attention mechanism is like a transcription factor that can "attend" to all accessible chromatin regions simultaneously, not just nearby ones—it can detect a distant enhancer at 50 kb as easily as one at 500 bp.

### The Problem with RNNs/LSTMs

Even LSTMs have limitations:

**1. Sequential processing:** Must process step 1 before step 2 before step 3...
- Can't parallelize → slow training on GPUs
- For 1000bp sequence, takes 1000 sequential operations

**2. Still limited range:** While better than RNNs, LSTMs struggle with dependencies 500+ steps apart
- An enhancer at position 1 affects a gene at position 800
- LSTM's memory weakens over long distances

**3. No direct "importance" mechanism:** LSTM learns what's important implicitly, but can't explicitly say "position 500 is crucial for position 800"

**Transformers solve all three problems with a revolutionary idea: attention.**

### The Core Idea: Attention Mechanism

Instead of reading a sequence step-by-step, **look at all positions at once and learn which ones are important for each other.**

**Analogy:** You're writing a research paper. Instead of reading all 50 references sequentially, you:
1. Skim all references simultaneously
2. Identify which ones are relevant for each section
3. Focus attention on relevant ones

**For genomics:** When processing position 800 (a promoter), the Transformer can directly look at position 500 (an enhancer) without reading through positions 501-799!

### How Attention Works (Simplified)

**The key insight:** Each position asks "Who should I pay attention to?" and gets answers from all other positions.

**Three simple steps:**

**1. Each position announces what it has:**
```
Position 500: "I'm an enhancer with GATA binding site"
Position 800: "I'm a promoter for gene X"
```

**2. Each position asks what it needs:**
```
Position 800 asks: "Where are my enhancers?"
Position 500 responds: "I'm an enhancer!" (high relevance)
Position 300 responds: "I'm in a coding region" (low relevance)
```

**3. Each position gets information from relevant positions:**
```
Position 800: Pays 70% attention to position 500 (enhancer)
              Pays 5% attention to position 300 (coding)
              Pays 25% to other positions
```

**The beautiful part:** The Transformer learns which positions are relevant through training!

> **[선택: 수식으로 보면]**
>
> For each position, three vectors are computed: Query (Q), Key (K), Value (V).
>
> Attention score: score(Q, K) = QKᵀ / √d
>
> Attention weights: α = softmax(score)
>
> Output: z = α · V
>
> The Query represents "what I'm looking for," the Key represents "what I have to offer," and the Value is the actual information passed. The division by √d prevents very large dot products.

### Visualizing Attention

When you train a Transformer, you can visualize attention weights to see what the model learned.

**Example reading:** "When processing the promoter at position 800, the model shows high attention weight (~1.0) at position 501, indicating this position is highly relevant."

**In biology, this might reveal:** Position 501 is an enhancer that regulates the gene at position 800—discovered by the model without being told about such relationships.

### Multi-Head Attention: Multiple Perspectives

Different biological relationships exist simultaneously:
- Enhancer-promoter interactions
- Transcription factor binding sites
- Chromatin loop anchors

**Solution:** Use multiple "attention heads," each learning different relationships:

```
Head 1: Learns enhancer-promoter pairs
Head 2: Learns transcription factor sites
Head 3: Learns splice site pairs
Head 4: Learns chromatin structure
...
```

**It's like having multiple experts, each specializing in different types of genomic relationships!**

### Key Advantages of Transformers

**1. Parallel processing:** Look at all positions simultaneously → fast training

**2. Long-range dependencies:** Position 1 can directly attend to position 1000 → no distance limit

**3. Interpretability:** Visualize attention weights → see what the model learned

**Connection to Chapters 2-3:** Remember Bayesian inference from Chapter 2? Attention is like computing "how much does this piece of evidence matter for my prediction?" The Transformer learns these relevance weights automatically!

### Real Applications: Enformer

Enformer uses Transformers to predict gene expression from DNA sequence ([Avsec et al 2021, Nature Methods](https://doi.org/10.1038/s41592-021-01252-x)):

**Task:** Given 200kb of DNA sequence, predict:
- Gene expression in 5,313 experiments
- Chromatin accessibility
- Histone modifications

**Architecture:**
- Convolutional layers (extract local features)
- **Transformer blocks** (capture long-range interactions)
- Can see relationships across entire 200kb!

**Key discovery:** The model learned that enhancers 50-100kb away regulate genes—exactly matching experimental Hi-C data! This demonstrates that Transformers can discover long-range regulatory relationships directly from sequence.

**Why it matters:** When you find a disease variant, Enformer can predict whether it affects a distant gene's expression, even without doing experiments.

### When to Use Transformers

✅ **Good for:**
- Long sequences (1000bp to 200kb+)
- Long-range dependencies (enhancers, 3D structure)
- When you have lots of data
- When you need interpretability (attention maps)

❌ **Not ideal for:**
- Small datasets (Transformers are data-hungry)
- Very short sequences (<100bp)
- Limited computational resources (Transformers are expensive)
- Simple local patterns (CNN is more efficient)

---

## 4.5 Choosing the Right Architecture

### Decision Framework

**Ask yourself these questions:**

**1. What structure does my data have?**
- Local spatial patterns (motifs in sequence) → **CNN**
- Sequential with temporal order (time series) → **RNN/LSTM**  
- Long-range dependencies (distant regulatory elements) → **Transformer**
- No special structure (tabular data) → **Fully-connected**

**2. How long are the sequences/dependencies?**
- Short (<50 positions) → **CNN**
- Medium (50-500 positions) → **LSTM**
- Long (500+ positions) → **Transformer**

**3. How much data do you have?**
- Small (<10,000 samples) → **CNN or small LSTM**
- Large (>100,000 samples) → **Transformer**

**4. Do you need interpretability?**
- Yes → **Transformer** (attention maps) or **CNN** (filter visualization)
- Less important → **Any architecture**

### Quick Reference Table

| Your Task | Recommended Architecture | Why |
|-----------|-------------------------|-----|
| Find TF binding motifs | CNN | Local patterns, position-independent |
| Predict splice sites | Bidirectional LSTM or Transformer | Need context from both directions |
| Classify cell types from expression | Fully-connected | No spatial/temporal structure |
| Predict next time point in trajectory | LSTM | Sequential dependencies |
| Find enhancer-promoter pairs | Transformer | Long-range dependencies (10-100kb) |
| Segment cells in microscopy | CNN (U-Net) | Spatial image patterns |
| Predict protein structure contacts | Transformer | Long-range residue interactions |

### Hybrid Architectures Often Work Best

Real-world problems often combine architectures:

**CNN + Transformer (like Enformer):**
```
CNN extracts local features (motifs)
    ↓
Transformer captures long-range interactions (regulatory elements)
    ↓
Final prediction
```

**Why hybrid?** CNNs are efficient for local patterns, Transformers excel at long-range. Use each for what it's good at!

### Practical Advice

**Start simple, then add complexity:**
1. Try a simple CNN first
2. If it doesn't capture all patterns, add LSTM or Transformer layers
3. If still not enough, try ensemble (combine multiple models)

**Don't over-engineer:**
- For 100bp sequences, CNN is probably enough
- For 10kb with clear long-range effects, go straight to Transformer
- For time series with <20 time points, simple LSTM works

---

## Summary

### Key Concepts

**1. Different architectures for different data structures:**
- **CNNs:** Sliding filters detect local patterns (motifs, features)
- **RNNs/LSTMs:** Sequential processing with memory (time series, trajectories)
- **Transformers:** Attention mechanism for long-range dependencies (regulatory elements)

**2. Why these architectures matter:**
- Match architecture to data structure → better performance
- Wrong architecture → can't learn important patterns
- Right architecture → learns efficiently with less data

**3. Trade-offs:**
- CNNs: Fast, efficient, but limited range
- LSTMs: Handle sequences well, but slow and limited range
- Transformers: Long-range, interpretable, but expensive

**4. Practical guidance:**
- Start with simpler architectures
- Use hybrid approaches for complex problems
- Consider computational constraints
- Visualize what models learn (filters, attention)

**5. Connection to foundations:**
- All use gradient descent (Chapter 3)
- All perform Bayesian updating (Chapter 2)
- Architecture determines what patterns they can learn

---

<details>
<summary><strong>📖 Key Terms</strong></summary>

| Term | Definition |
|------|-----------|
| **Convolutional Neural Network (CNN)** | Architecture using sliding filters to detect local patterns |
| **Filter/Kernel** | Small matrix of weights that slides across input |
| **Feature map** | Output showing where patterns were detected |
| **Pooling** | Summarizing/reducing spatial dimensions |
| **Recurrent Neural Network (RNN)** | Architecture processing sequences step-by-step |
| **Hidden state** | RNN's memory carried forward across time steps |
| **Vanishing gradient** | Problem where gradients become too small to enable learning |
| **LSTM** | RNN variant with gates to control memory |
| **Gates** | Mechanisms controlling information flow (forget, input, output) |
| **Transformer** | Architecture using attention for long-range dependencies |
| **Attention** | Mechanism for weighting importance of different positions |
| **Multi-head attention** | Multiple attention mechanisms learning different relationships |
| **Bidirectional** | Processing sequence in both forward and backward directions |
| **Receptive field** | Range of input that affects a particular output |

</details>

---

## Test Your Understanding

1. **Architecture intuition:** You need to predict whether a 500bp DNA sequence is an active enhancer. The key features are: (1) presence of TF binding motifs (8-15bp each) and (2) specific combinations of motifs within 100bp windows. Would you use CNN, LSTM, or Transformer? Why?

2. **CNNs and weight sharing:** Explain why using the same filter at every position (weight sharing) is useful for finding motifs. What would happen if each position had its own unique filter?

3. **Memory and biology:** LSTMs can "remember" information across long sequences. Give two biological examples where this long-term memory would be crucial for making accurate predictions.

4. **Attention visualization:** If you visualize attention weights from a Transformer trained on gene regulation, and you see strong attention between position 1000 and position 50000, what biological relationship might this reveal? How would you experimentally validate this?

5. **Comparing approaches:** You have three models for splice site prediction:
   - Model A (CNN): 95% accurate, trains in 2 hours
   - Model B (LSTM): 96% accurate, trains in 12 hours
   - Model C (Transformer): 97% accurate, trains in 5 hours, but needs 10x more training data
   
   Which would you choose if you have: (a) limited data, (b) limited time, (c) need best accuracy regardless of cost?

6. **Hybrid reasoning:** Enformer uses CNNs followed by Transformers. Why not just use Transformers from the start? What advantage does the CNN provide?

7. **Bidirectional context:** When would you want bidirectional processing instead of forward-only? Give a specific biological example where information from both directions matters.

8. **Architecture limitations:** What problems would arise if you tried to use a CNN to predict gene expression 24 hours from now, given measurements from hours 0, 6, 12, 18? What about using an LSTM?

---

## Further Reading

### Foundational Papers

**CNNs for Genomics:**
- **Zhou & Troyanskaya (2015)** "Predicting effects of noncoding variants with deep learning—based sequence model." *Nature Methods* 12:931-934.
  - DeepSEA: First major success of CNNs for regulatory genomics
  - [DOI: 10.1038/nmeth.3547](https://doi.org/10.1038/nmeth.3547)

**Transformers:**
- **Vaswani et al. (2017)** "Attention is all you need." *NeurIPS* 2017.
  - The original Transformer paper
  - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- **Avsec et al. (2021)** "Effective gene expression prediction from sequence by integrating long-range interactions." *Nature Methods* 18:1196-1203.
  - Enformer: Transformer for genomics with 200kb context
  - [DOI: 10.1038/s41592-021-01252-x](https://doi.org/10.1038/s41592-021-01252-x)

### Reviews

- **Eraslan et al. (2019)** "Deep learning: new computational modelling techniques for genomics." *Nature Reviews Genetics* 20:389-403.
  - [DOI: 10.1038/s41576-019-0122-6](https://doi.org/10.1038/s41576-019-0122-6)

- **Zou et al. (2019)** "A primer on deep learning in genomics." *Nature Genetics* 51:12-18.
  - [DOI: 10.1038/s41588-018-0295-5](https://doi.org/10.1038/s41588-018-0295-5)

### Online Resources

- **The Illustrated Transformer** by Jay Alammar
  - [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)

- **Distill.pub: Attention and Augmented RNNs**
  - [https://distill.pub/2016/augmented-rnns/](https://distill.pub/2016/augmented-rnns/)

---

## What's Next?

You now understand the three major neural network architectures and when to use each!

**You've learned:**
- CNNs detect local patterns efficiently  
- LSTMs handle sequential dependencies with memory  
- Transformers capture long-range interactions through attention  
- How to choose the right architecture for biological problems

**Before moving on, make sure you can:**
- [ ] Explain how CNNs use filters to detect motifs
- [ ] Describe why LSTMs solve the vanishing gradient problem
- [ ] Explain attention mechanism in simple terms
- [ ] Choose appropriate architecture for a biological problem

**Self-check:**
1. What's the key advantage of CNNs over fully-connected networks for sequence data?
2. Why can't basic RNNs remember patterns from 100 steps ago?
3. How do Transformers "see" long-range dependencies?
4. When would you use a hybrid CNN+Transformer architecture?

👉 **[Continue to Chapter 5: Genetic Variation and Genomic Technologies](chapter5.md)**

---

*"The future of biology is at the intersection of experiments and computation. Neither alone is sufficient."*
