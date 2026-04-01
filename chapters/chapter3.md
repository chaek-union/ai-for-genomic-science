# Chapter 3: Neural Networks — How Machines Learn Patterns

**[Interactive Lab: Chapter 3](https://chaek-union.github.io/ai-for-genomic-science/interactive/chapter3.html)** — 2D 분류기, 활성화 함수 비교, 단일 뉴런 탐색기, 학습 과정 시각화를 직접 체험해보세요!

You're a second-year biology student staring at 200 DNA sequences, each 20 nucleotides long. Half of them are real splice donor sites (the GT at exon-intron boundaries where your pre-mRNA gets cut), and half are decoy sequences that look similar but aren't functional. Your professor challenges you: *"Can you figure out the rule that separates real splice sites from fake ones?"*

You start by looking for the obvious pattern: **GT** at positions 3-4. Sure enough, almost every real splice site has it. But so do most of the decoys—your professor was sneaky. You look more carefully. Real ones tend to have a **G** or **A** at position 2, a run of purines before the GT, and a specific pattern at positions 5-8. You start combining rules: "If position 2 is A AND position 5 is A AND position 7 is G, then... probably real?"

After an hour, you have a messy decision tree with 12 rules that correctly classifies about 75% of the sequences. Your professor smiles and says: *"Not bad. But a neural network can learn to do this at 95% accuracy in about 3 seconds—and it discovers patterns you didn't even notice."*

**How?**

That's what this chapter is about. In Chapter 2, you learned that deep learning is a practical approximation of Bayesian inference—finding the best explanation (MAP) by minimizing a loss function. Now we'll open the box and see the actual machinery: **artificial neurons**, **layers**, **forward propagation**, and the remarkable algorithm called **backpropagation** that allows networks to learn from mistakes.

The key insight: neural networks aren't magic. They're built from the simplest possible building block—a single artificial neuron that does weighted addition followed by a simple nonlinear function. Stack enough of these together, and they can learn almost any pattern.

---

## From Biological Neurons to Artificial Neurons

### The Real Thing: Your Brain's Neurons

You learned about neurons in introductory biology. Let's revisit the key features:

**A biological neuron:**
1. **Receives inputs** — dendrites collect signals from thousands of other neurons
2. **Integrates** — the cell body sums up excitatory and inhibitory signals
3. **Thresholds** — if the total signal exceeds a threshold, the neuron fires
4. **Transmits** — the axon sends the signal to downstream neurons

```
Dendrites (inputs) → Cell body (integration + threshold) → Axon (output)
     ↑                         ↑                              ↑
 Signals from          Sum of signals              Signal to next
 other neurons        exceeds threshold?            neurons
```

**Key properties:**
- Each input connection has a different **strength** (synaptic weight)
- Frequently used connections get **stronger** (long-term potentiation)
- Rarely used connections get **weaker** (long-term depression)
- The neuron's response is **nonlinear** — it either fires or doesn't (approximately)

This is exactly what artificial neurons mimic.

### The Artificial Version: A Single Neuron (Perceptron)

An artificial neuron takes the biological concept and simplifies it to pure math:

```
Inputs:      x₁, x₂, x₃, ... xₙ     (like signals from dendrites)
Weights:     w₁, w₂, w₃, ... wₙ     (like synaptic strengths)
Bias:        b                         (like the resting potential)
Summation:   z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b    (like cell body integration)
Activation:  a = f(z)                  (like the firing threshold)
Output:      a                         (like the axon signal)
```

**Genomics example: Is this variant pathogenic?**

Imagine a single neuron that takes three features of a genetic variant:
- x₁ = Conservation score (0 to 1, how conserved is this position?)
- x₂ = Population frequency (0 to 1, how common is this variant?)
- x₃ = Predicted structural impact (0 to 1, does it change protein structure?)

```
z = (0.8 × conservation) + (-0.6 × frequency) + (0.5 × structural_impact) + (-0.1)
      ↑                        ↑                       ↑                       ↑
  "High conservation       "Common variants        "Structural changes       Bias
   suggests important"     are usually benign"      may be damaging"
```

Notice:
- **w₁ = 0.8** (positive, large): high conservation → more likely pathogenic
- **w₂ = -0.6** (negative): high frequency → less likely pathogenic
- **w₃ = 0.5** (positive, moderate): structural change → somewhat more likely pathogenic
- **b = -0.1** (small negative): slight default toward "benign"

**The weights encode what the neuron has learned about the relationship between inputs and the output.**

### The Analogy Table

| Biological Neuron | Artificial Neuron | What It Does |
|-------------------|-------------------|--------------|
| Dendrites | Inputs (x₁, x₂, ...) | Receive information |
| Synaptic strength | Weights (w₁, w₂, ...) | How much each input matters |
| Resting potential | Bias (b) | Default tendency |
| Cell body summation | z = Σwᵢxᵢ + b | Combine all information |
| Firing threshold | Activation function f(z) | Decide whether to "fire" |
| Axon output | Output a = f(z) | Send signal forward |
| Long-term potentiation | Increasing weight during training | Learning from experience |
| Long-term depression | Decreasing weight during training | Unlearning wrong patterns |

**Important caveat:** This analogy is useful for intuition, but artificial neurons are a dramatic simplification. Real neurons communicate through complex temporal patterns of spikes, use dozens of neurotransmitters, and have intricate dendritic computation. The analogy helps you understand the *concept*, not the biology.

---

## Activation Functions: Why Nonlinearity Matters

### The Problem with Just Adding Things Up

Without an activation function, our neuron just computes: z = w₁x₁ + w₂x₂ + b

This is a **linear function**. It can only draw straight lines.

**Why is this a problem?**

Think about classifying genetic variants as pathogenic vs. benign. In reality:
- A variant with **moderate** conservation AND **moderate** structural change might be pathogenic
- But **low** conservation with **high** structural change might be benign (it's a variable region)
- And **high** conservation with **low** structural change might also be benign (it's a synonymous variant in a conserved gene)

This relationship is **nonlinear**—you can't separate pathogenic from benign with a single straight line in the conservation × structural_impact space.

**The activation function adds the nonlinearity we need.**

### The Classic Three

#### 1. Sigmoid: The Smooth Switch

```
σ(z) = 1 / (1 + e^(-z))

Input:  any number from -∞ to +∞
Output: a number between 0 and 1
```

**Biology analogy:** Like the dose-response curve in pharmacology. At low drug concentrations, there's no response. At high concentrations, the response saturates. In between, there's a smooth S-shaped transition.

**When it's useful:** When you want an output that represents a probability. "There's a 73% chance this variant is pathogenic."

**The problem:** For very large or very small inputs, the sigmoid is nearly flat (gradient ≈ 0). This makes learning very slow — a problem called **vanishing gradients** that we'll discuss later.

#### 2. Tanh: The Centered Switch

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

Input:  any number from -∞ to +∞
Output: a number between -1 and +1
```

**Biology analogy:** Like gene expression fold-change centered on zero. Positive means upregulated, negative means downregulated, zero means no change. The magnitude tells you how strong the effect is, saturating at extreme values.

**Advantage over sigmoid:** Centered around zero, which helps with training. But still suffers from vanishing gradients at the extremes.

#### 3. ReLU: The Simple Gate

```
ReLU(z) = max(0, z)

Input:  any number
Output: 0 if input is negative, otherwise the input itself
```

**Biology analogy:** Like a gene that's either silent (expression = 0) or active (expression proportional to signal). There's no "negative expression" — once the gene is off, it's off. But when it's on, the response is proportional to the input.

**Why it's the most popular today:**
- Extremely simple and fast to compute
- No vanishing gradient problem (for positive inputs)
- Sparse activation — many neurons output exactly 0, which is computationally efficient

**The downside:** "Dead neurons" — if a neuron's input is always negative, it always outputs 0 and can never recover. Variants like **Leaky ReLU** (allows a small negative slope) fix this.

### Which One to Use?

| Activation | Output Range | Best For | Problem |
|-----------|-------------|----------|---------|
| Sigmoid | 0 to 1 | Output layer (binary classification) | Vanishing gradients |
| Tanh | -1 to 1 | Hidden layers (old models) | Vanishing gradients |
| ReLU | 0 to ∞ | Hidden layers (modern default) | Dead neurons |

**Modern practice:** Use ReLU (or its variants) for hidden layers, sigmoid for binary output, softmax (generalized sigmoid) for multi-class output.

---

## A Single Neuron as a Classifier

### Drawing a Line in Variant Space

Let's see what a single neuron can actually do. With two inputs (conservation score and structural impact), the neuron computes:

```
z = w₁ × conservation + w₂ × structural_impact + b
output = sigmoid(z)
```

If output > 0.5, we predict "pathogenic." Otherwise, "benign."

**The decision boundary** is where output = 0.5, which means z = 0:

```
w₁ × conservation + w₂ × structural_impact + b = 0
```

This is the equation of a **straight line** in 2D space (or a plane in 3D, or a hyperplane in higher dimensions).

**The single neuron divides the input space with a straight line.**

Everything on one side is predicted pathogenic; everything on the other side is predicted benign.

### When One Line Isn't Enough

What if the real pattern looks like this?

```
                    Structural Impact
                    ↑
                    |  B B B B P P
                    |  B B P P P P
                    |  B P P P P B
                    |  P P P B B B
                    |  P P B B B B
                    +——————————————→ Conservation
                    
P = Pathogenic, B = Benign
```

The pathogenic variants form a diagonal band — you can't separate them with a single straight line! This is an example of a **nonlinear decision boundary**.

**This is why we need multiple neurons organized in layers.**

---

## Building a Network: Layers of Neurons

### From One Neuron to Many

A single neuron draws one line. What if we combine multiple neurons?

```
Layer 1: Three neurons, each drawing a different line

Neuron 1: "Is conservation > 0.7?"
Neuron 2: "Is structural impact > 0.5?"  
Neuron 3: "Is conservation + structural impact > 1.0?"

Layer 2: One neuron that combines the answers

Neuron 4: "Based on answers from neurons 1-3, is this pathogenic?"
```

**Three lines can carve out a triangular region in the input space.** More neurons = more lines = more complex boundaries.

### The Architecture

```
Input Layer        Hidden Layer         Output Layer
(features)        (learned patterns)    (prediction)

  x₁ ——→  [h₁]  ——→
       ╲  ╱   ╲       
  x₂ ——→  [h₂]  ——→  [output]  →  "Pathogenic" or "Benign"
       ╱  ╲   ╱
  x₃ ——→  [h₃]  ——→
```

**Input layer:** Raw features (conservation, frequency, structural impact)

**Hidden layer(s):** Neurons that discover intermediate patterns. We don't tell them what to look for — they figure it out during training.

**Output layer:** Final prediction.

**Each connection has its own weight.** In this small network:
- Input → Hidden: 3 inputs × 3 hidden neurons = 9 weights + 3 biases = 12 parameters
- Hidden → Output: 3 hidden × 1 output = 3 weights + 1 bias = 4 parameters
- **Total: 16 parameters** that the network learns

Modern networks like AlphaFold have **millions** of parameters. The principle is the same — just more neurons and more layers.

### Why "Deep" Learning?

A network with many hidden layers is called a **deep neural network**:

```
Input → Hidden 1 → Hidden 2 → Hidden 3 → ... → Output
         ↑            ↑           ↑
     Simple        Intermediate   Complex
     patterns      combinations   abstractions
```

**Genomics analogy: How a deep network reads DNA**

```
Layer 1: Detects simple motifs
         "There's a TATA at this position"
         "There's a GC-rich region here"

Layer 2: Combines motifs into modules
         "TATA box + Inr element = core promoter signature"
         "CpG island + GC-box = housekeeping promoter pattern"

Layer 3: Recognizes regulatory logic
         "Core promoter + enhancer motifs within 500bp = active promoter"
         "Core promoter + repressor motifs = silenced promoter"

Layer 4: Predicts function
         "This sequence is an active enhancer in neural tissue"
```

Each layer builds on the previous one, creating increasingly abstract representations. The first layer sees individual nucleotides; the last layer understands regulatory function.

**This hierarchical feature learning is what makes deep learning so powerful for genomics.**

---

## Forward Propagation: How Information Flows

### Step by Step

**Forward propagation** is simply computing the output of the network given an input. Information flows forward from input to output, one layer at a time.

**Example: Predicting splice site functionality**

Input: A 20-nucleotide DNA sequence, one-hot encoded (each position is 4 numbers: [A, C, G, T])
- Total input: 20 × 4 = 80 numbers

```
Step 1: Input → Hidden Layer 1 (10 neurons)
  For each hidden neuron j (j = 1 to 10):
    z_j = w₁ⱼ×x₁ + w₂ⱼ×x₂ + ... + w₈₀ⱼ×x₈₀ + bⱼ
    h_j = ReLU(z_j)
  
  Result: 10 numbers representing "features" the network discovered

Step 2: Hidden Layer 1 → Hidden Layer 2 (5 neurons)
  For each neuron k (k = 1 to 5):
    z_k = w₁ₖ×h₁ + w₂ₖ×h₂ + ... + w₁₀ₖ×h₁₀ + bₖ
    h_k = ReLU(z_k)
  
  Result: 5 numbers representing higher-level patterns

Step 3: Hidden Layer 2 → Output (1 neuron)
  z_out = w₁×h₁ + w₂×h₂ + ... + w₅×h₅ + b
  output = sigmoid(z_out)
  
  Result: 0.87 → "87% likely to be a functional splice site"
```

**That's it.** Forward propagation is just repeated weighted sums followed by activation functions. Matrix multiplication and simple nonlinear functions — nothing more.

### One-Hot Encoding: How DNA Becomes Numbers

Neural networks need numerical inputs. How do we convert a DNA sequence?

```
Sequence:  A  T  G  C  A  ...
           ↓  ↓  ↓  ↓  ↓
    A:     1  0  0  0  1  ...
    C:     0  0  0  1  0  ...
    G:     0  0  1  0  0  ...
    T:     0  1  0  0  0  ...
```

Each nucleotide becomes a vector of length 4, with a 1 at the position corresponding to the base and 0s elsewhere. A 20-nucleotide sequence becomes 80 numbers — a format the network can process.

**Why not just use A=1, C=2, G=3, T=4?** Because that would imply G is "greater than" C or that T-G = A, which makes no biological sense. One-hot encoding treats each base as an independent category.

---

## Loss Functions: Measuring How Wrong We Are

### Connecting to Chapter 2

Remember from Chapter 2:

```
Minimizing loss = Maximizing likelihood
Loss = -log P(data | parameters)
```

Now let's see this in practice with neural networks.

### Binary Cross-Entropy: The Standard for Classification

For predicting "pathogenic or benign":

```
Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

Where:
  y = true label (1 = pathogenic, 0 = benign)
  ŷ = network's prediction (e.g., 0.87)
```

**Let's compute:**

**Example 1: Good prediction**
- True: pathogenic (y = 1)
- Predicted: 0.92
- Loss = -[1 × log(0.92) + 0 × log(0.08)] = -log(0.92) = 0.083

**Example 2: Bad prediction**
- True: pathogenic (y = 1) 
- Predicted: 0.15
- Loss = -[1 × log(0.15) + 0 × log(0.85)] = -log(0.15) = 1.897

**Example 3: Confident wrong prediction**
- True: benign (y = 0)
- Predicted: 0.98 (confidently says pathogenic!)
- Loss = -[0 × log(0.98) + 1 × log(0.02)] = -log(0.02) = 3.912

**Key insight:** The loss is small when the prediction matches the truth, and large when it doesn't. Being **confidently wrong** is penalized most severely.

### For Entire Training Set

```
Total Loss = (1/N) × Σ Loss for each training example

"Average badness across all training examples"
```

**Training goal:** Find the weights that minimize this total loss.

---

## Backpropagation: How Networks Learn

### The Key Question

We know:
- Forward propagation computes predictions
- Loss function tells us how wrong we are
- We need to adjust weights to reduce the loss

**But how do we know which weights to change, and by how much?**

This is where **backpropagation** comes in — arguably the most important algorithm in deep learning.

### The Intuition: Blame Assignment

Imagine a factory assembly line that produces defective products:

```
Raw materials → Station A → Station B → Station C → Final product (defective!)
                   ↑            ↑            ↑
              "Was it      "Or was it    "Or was it
               my fault?"   my fault?"    my fault?"
```

When the final product is bad, you need to figure out which station is responsible. You trace backward from the defect:

1. **Check Station C** (closest to the output): Did it contribute to the defect?
2. **Check Station B**: Did it pass bad intermediate products to C?
3. **Check Station A**: Did it start the chain of problems?

**Backpropagation does exactly this** — it traces backward from the loss, computing how much each weight contributed to the error.

### The Chain Rule: The Math Behind the Intuition

Backpropagation relies on a calculus concept called the **chain rule**:

```
If y depends on u, and u depends on x, then:

dy/dx = dy/du × du/dx

"How y changes with x" = "How y changes with u" × "How u changes with x"
```

**Applied to our network:**

```
How Loss changes with w₁ (a weight in Layer 1):

dLoss/dw₁ = dLoss/dOutput × dOutput/dHidden × dHidden/dw₁
               ↑                ↑                ↑
          "How much does   "How much does    "How much does
           output affect    hidden layer      w₁ affect
           the loss?"       affect output?"   hidden layer?"
```

Each factor is easy to compute individually. Multiplied together, they tell us exactly how much w₁ contributes to the loss — and therefore how to adjust it.

### The Complete Training Loop

```
Repeat for many epochs:
  
  For each batch of training examples:
    
    1. FORWARD PASS
       Input → Hidden → Output → Prediction
    
    2. COMPUTE LOSS
       Compare prediction to true label
       Loss = how wrong we are
    
    3. BACKWARD PASS (Backpropagation)
       Compute gradient of loss with respect to every weight
       "How much is each weight responsible for the error?"
    
    4. UPDATE WEIGHTS
       w_new = w_old - learning_rate × gradient
       "Adjust each weight to reduce the error"
```

**The learning rate** controls how big each adjustment step is:
- Too large: overshoots the optimal weights, loss oscillates or diverges
- Too small: converges very slowly, might get stuck
- Just right: steady progress toward the minimum loss

**Connecting to Chapter 2:** This is the gradient descent we discussed — climbing the posterior landscape (or equivalently, descending the loss landscape). Backpropagation is how we compute the direction to move.

---

## Putting It All Together: A Genomics Example

### Predicting Pathogenic Variants Step by Step

**Task:** Train a network to predict whether coding variants are pathogenic.

**Data:** 10,000 labeled variants from ClinVar
- Features: conservation score, population frequency, predicted structural impact, distance to active site, gene constraint (pLI score)
- Label: Pathogenic (1) or Benign (0)

**Network architecture:**

```
5 inputs → [16 ReLU neurons] → [8 ReLU neurons] → [1 sigmoid neuron]
                                                          ↓
                                                    P(pathogenic)
```

**Total parameters:** (5×16+16) + (16×8+8) + (8×1+1) = 96 + 136 + 9 = **241 parameters**

**Training process:**

```
Epoch 1:  Random weights → Loss = 0.693 (equivalent to random guessing)
Epoch 10: Learning patterns → Loss = 0.42
          Network discovers: "high conservation = more likely pathogenic"
Epoch 50: Refining → Loss = 0.18
          Network discovers: "rare + conserved + structural change = pathogenic"
Epoch 100: Converged → Loss = 0.12
          Network has learned complex nonlinear interactions
          "High conservation + rare + near active site" → strong pathogenic signal
          "High conservation + common" → probably benign (population evidence)
```

**Key observation:** We never told the network these rules. It **discovered them from data** by adjusting 241 parameters to minimize prediction errors.

---

## Common Pitfalls and How to Avoid Them

### Overfitting: Memorizing Instead of Learning

**The problem:** A network with enough parameters can memorize the training data perfectly — but fail on new data.

**Genomics analogy:** Imagine studying for an exam by memorizing every single practice problem with its exact answer. You score 100% on the practice set. But on the real exam, the questions are slightly different, and you fail because you never understood the underlying principles.

```
Training accuracy: 99.5%  ← Memorized training data
Test accuracy:     62.3%  ← Fails on new data

This network learned: "Variant rs12345 is pathogenic"
Instead of learning: "Rare, conserved variants in constrained genes tend to be pathogenic"
```

**Solutions:**

| Technique | What It Does | Biology Analogy |
|-----------|-------------|-----------------|
| **More training data** | Harder to memorize when there's more to learn | Studying from 100 textbooks vs. 1 |
| **Dropout** | Randomly disable neurons during training | Studying with a different study group each day |
| **Regularization** | Penalize large weights (Chapter 2's "prefer simpler models") | Occam's razor |
| **Early stopping** | Stop training before overfitting occurs | Knowing when to stop studying and sleep |
| **Validation set** | Monitor performance on unseen data | Practice exams that aren't in your study guide |

### Vanishing Gradients: When Deep Networks Can't Learn

**The problem:** In deep networks with sigmoid/tanh activations, gradients become extremely small in early layers, so they learn very slowly or not at all.

```
Layer 5 gradient: 0.25
Layer 4 gradient: 0.25 × 0.25 = 0.0625
Layer 3 gradient: 0.0625 × 0.25 = 0.0156
Layer 2 gradient: 0.0156 × 0.25 = 0.0039
Layer 1 gradient: 0.0039 × 0.25 = 0.00098  ← Almost zero! Layer 1 barely learns.
```

**Solutions:**
- **ReLU activation** — gradient is either 0 or 1 (no shrinking)
- **Residual connections** — "skip connections" that let gradients flow directly to early layers (used in modern architectures like ResNet and Transformers)
- **Batch normalization** — keeps intermediate values in a reasonable range

### Choosing Hyperparameters

**Hyperparameters** are choices YOU make before training (unlike weights, which the network learns).

| Hyperparameter | What to Try | Rule of Thumb |
|----------------|------------|---------------|
| Learning rate | 0.001, 0.01, 0.1 | Start with 0.001 for Adam optimizer |
| Number of layers | 1-5 for most tasks | Start simple, add layers if underfitting |
| Neurons per layer | 16, 32, 64, 128 | Wider for more complex data |
| Batch size | 32, 64, 128 | 32 is a good default |
| Epochs | 10-1000 | Use early stopping |

**The golden rule: Start simple.** A small network that works is better than a large network that overfits.

---

## Math Box: Forward and Backward Pass

*You can skip this section and still understand the concepts! This is for those curious about the formal mathematics.*

### Forward Pass (Matrix Form)

For a network with one hidden layer:

```
Hidden layer:
  z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾     (matrix multiplication + bias)
  h = f(z⁽¹⁾)               (activation function, element-wise)

Output layer:
  z⁽²⁾ = W⁽²⁾h + b⁽²⁾
  ŷ = σ(z⁽²⁾)               (sigmoid for binary classification)
```

Where:
- W⁽¹⁾ is the weight matrix connecting input to hidden layer
- W⁽²⁾ is the weight matrix connecting hidden to output layer
- f is the activation function (e.g., ReLU)
- σ is the sigmoid function

### Backward Pass (Gradient Computation)

Starting from the loss L = -[y log(ŷ) + (1-y) log(1-ŷ)]:

```
Step 1: Gradient at output
  dL/dz⁽²⁾ = ŷ - y         (remarkably simple!)

Step 2: Gradient for output weights
  dL/dW⁽²⁾ = (ŷ - y) × hᵀ
  dL/db⁽²⁾ = ŷ - y

Step 3: Gradient passed to hidden layer
  dL/dh = W⁽²⁾ᵀ × (ŷ - y)

Step 4: Gradient through activation
  dL/dz⁽¹⁾ = dL/dh ⊙ f'(z⁽¹⁾)    (⊙ = element-wise multiplication)
  
  For ReLU: f'(z) = 1 if z > 0, else 0

Step 5: Gradient for hidden weights
  dL/dW⁽¹⁾ = dL/dz⁽¹⁾ × xᵀ
  dL/db⁽¹⁾ = dL/dz⁽¹⁾
```

### Weight Update (Gradient Descent)

```
W⁽¹⁾ ← W⁽¹⁾ - α × dL/dW⁽¹⁾
b⁽¹⁾ ← b⁽¹⁾ - α × dL/db⁽¹⁾
W⁽²⁾ ← W⁽²⁾ - α × dL/dW⁽²⁾
b⁽²⁾ ← b⁽²⁾ - α × dL/db⁽²⁾

Where α = learning rate
```

The beauty of backpropagation: no matter how deep the network, the chain rule lets us compute gradients for every weight efficiently in a single backward pass.

---

## Summary

**Key Takeaways:**

1. **Artificial neurons mimic biological neurons** — weighted inputs, summation, nonlinear activation, output. But this is a simplification, not a biological model.

2. **Activation functions add nonlinearity** — without them, any network collapses to a single linear function. ReLU is the modern default for hidden layers.

3. **A single neuron draws a straight line** — it can only separate data that's linearly separable. This is why we need multiple neurons and layers.

4. **Layers build hierarchical features** — early layers detect simple patterns (DNA motifs), later layers combine them into complex concepts (regulatory logic).

5. **Forward propagation** is repeated weighted sums + activations. Information flows from input to output.

6. **Loss functions measure prediction error** — binary cross-entropy for classification, connecting to Chapter 2's negative log-likelihood.

7. **Backpropagation computes gradients** using the chain rule — it assigns "blame" to each weight for the error, working backward from the output.

8. **Training = forward pass + loss + backward pass + weight update**, repeated thousands of times.

9. **Overfitting is the main danger** — combat it with more data, dropout, regularization, and early stopping.

10. **Start simple** — a small working network is better than a complex failing one.

---

## Key Terms

- **Artificial neuron (perceptron):** Basic unit that computes a weighted sum of inputs, adds bias, applies activation function
- **Weight:** Learned parameter controlling the strength of a connection between neurons
- **Bias:** Learned parameter allowing the neuron's activation threshold to shift
- **Activation function:** Nonlinear function applied after summation (ReLU, sigmoid, tanh)
- **ReLU (Rectified Linear Unit):** max(0, z) — most popular modern activation
- **Hidden layer:** Layer of neurons between input and output that learns intermediate features
- **Deep neural network:** Network with multiple hidden layers
- **Forward propagation:** Computing output from input through the network layers
- **One-hot encoding:** Representing categorical data (A, C, G, T) as binary vectors
- **Loss function:** Measures how wrong the network's predictions are
- **Binary cross-entropy:** Standard loss for binary classification problems
- **Backpropagation:** Algorithm that computes gradients of the loss with respect to all weights using the chain rule
- **Gradient:** Direction and magnitude of steepest increase in a function
- **Learning rate:** Hyperparameter controlling the step size of weight updates
- **Epoch:** One complete pass through the entire training dataset
- **Overfitting:** Network memorizes training data instead of learning general patterns
- **Dropout:** Regularization technique that randomly disables neurons during training
- **Vanishing gradient:** Problem where gradients become too small for early layers to learn
- **Hyperparameter:** Settings chosen before training (learning rate, architecture, etc.)

---

## Test Your Understanding: Can You Answer These?

<details>
<summary><strong>1. Why can't a single neuron classify XOR (exclusive or) patterns? And why does this matter for genomics?</strong></summary>

**Answer:**

XOR is a pattern where the output is 1 only when exactly one of two inputs is 1:

```
Input A | Input B | Output
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
```

A single neuron can only draw a straight line to separate classes. But XOR requires a curved or multi-part boundary — no single line can separate the 1s from the 0s.

**Why this matters for genomics:** Many biological relationships are XOR-like:
- A variant might be pathogenic when it occurs in Gene A **alone** OR Gene B **alone**, but benign when both are mutated (because the second mutation is compensatory)
- A drug might work when Pathway X is active OR Pathway Y is active, but fail when both are active (toxic overdose)

These nonlinear interactions require **at least two layers** (a hidden layer + output layer) to capture. This is one of the fundamental reasons we need deep networks for biological prediction tasks.

</details>

<details>
<summary><strong>2. A researcher trains a variant pathogenicity predictor on European-ancestry data. It achieves 94% accuracy on European test data but only 71% on African-ancestry data. Explain what happened using concepts from this chapter.</strong></summary>

**Answer:**

This is **overfitting to the training distribution**.

During training, the network adjusted its 241+ parameters to minimize loss on European-ancestry variants. The patterns it learned are:
- Which population frequencies are "rare" (calibrated to European allele frequencies)
- Which conservation patterns correlate with pathogenicity (in European-studied genes)
- Which structural changes are damaging (based on European disease associations)

**The problem:** African genomes have ~25% more genetic variation than European genomes. Many variants that are common in African populations are rare or absent in European databases. The network has never seen these patterns during training, so it misclassifies them — often flagging normal African variation as "pathogenic" because it looks rare from a European perspective.

**This is not just overfitting — it's a dangerous form of bias.** Solutions include:
- Training on diverse, multi-ancestry datasets
- Using a validation set that includes underrepresented populations
- Reporting per-ancestry performance metrics, not just overall accuracy
- Using population-specific frequency filters

**From a neural network perspective:** The network's learned weights encode European-specific patterns. The decision boundaries it drew during training don't generalize to regions of feature space occupied by African-ancestry variants.

</details>

<details>
<summary><strong>3. Your network for predicting enhancer activity has training loss of 0.08 but validation loss of 0.95. What's happening, and what would you try?</strong></summary>

**Answer:**

**Diagnosis:** Severe overfitting. The network has memorized the training enhancers (low training loss) but cannot generalize to new enhancers (high validation loss).

**The gap:**
```
Training loss:   0.08  (excellent on seen data)
Validation loss: 0.95  (terrible on unseen data)
Gap:             0.87  (huge — clear overfitting)
```

**What to try, in order:**

1. **More training data** — If you have only 500 enhancer sequences, try to get 5,000. This is often the single most effective fix.

2. **Regularization** — Add L2 regularization (weight decay) to penalize large weights. Start with λ = 0.01.

3. **Dropout** — Add dropout (p = 0.3-0.5) after hidden layers. This forces the network to learn redundant representations.

4. **Simpler architecture** — Reduce the number of layers and neurons. If you have a 5-layer network, try 2 layers. The network might be too complex for the amount of data you have.

5. **Early stopping** — Monitor validation loss during training and stop when it starts increasing (even if training loss is still decreasing).

6. **Data augmentation** — For DNA sequences, you could augment by reverse-complementing sequences (both strands should have similar regulatory activity).

**The underlying principle from Chapter 2:** With limited data, the posterior distribution is wide (high uncertainty). A complex model with many parameters can find a sharp peak in training data that doesn't generalize. Regularization acts like an informative prior, keeping the model simpler and more generalizable.

</details>

---

## Hands-On Labs

### Lab 3.1: Build Your First Neural Network (45-60 min)

**Learn:**
- Implement a single neuron from scratch in Python
- Visualize decision boundaries
- See how activation functions change the boundary
- Train on a simple genomic classification task

**[Access Lab 3.1 on Google Colab](https://colab.research.google.com/drive/YOUR_LAB3_1_LINK_HERE)**

### Lab 3.2: Train a Variant Pathogenicity Predictor (60-90 min)

**Learn:**
- Build a multi-layer network with PyTorch
- Train on real ClinVar data
- Visualize training loss over epochs
- Diagnose and fix overfitting
- Evaluate with confusion matrices and ROC curves

**[Access Lab 3.2 on Google Colab](https://colab.research.google.com/drive/YOUR_LAB3_2_LINK_HERE)**
