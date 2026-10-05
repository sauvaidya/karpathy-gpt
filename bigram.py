import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. Read training data
# ---------------------------------------------------------

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text)


# ---------------------------------------------------------
# 2. Build vocabulary
# ---------------------------------------------------------

chars = sorted(set(text))
vocab_size = len(chars)

print("Vocabulary:")
print(chars)

print("Vocabulary size:", vocab_size)


# ---------------------------------------------------------
# 3. Character <-> ID mappings
# ---------------------------------------------------------

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print("Character to ID:")
print(stoi)

print("ID to character:")
print(itos)


# ---------------------------------------------------------
# 4. Encode / decode
# ---------------------------------------------------------

def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return ''.join(itos[i] for i in ids)


print("Encoded 'cat':")
print(encode("cat"))

print("Decoded:")
print(decode(encode("cat")))


# ---------------------------------------------------------
# 5. Convert text to tensor
# ---------------------------------------------------------

data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# ---------------------------------------------------------
# 6. Configuration
# ---------------------------------------------------------

block_size = 8
batch_size = 4

n_embd = 16

num_heads = 4

head_size = n_embd // num_heads

n_layer = 2

learning_rate = 0.01

max_iters = 200


# ---------------------------------------------------------
# 7. Batch creation
# ---------------------------------------------------------

def get_batch():

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    X = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    Y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return X, Y


# ---------------------------------------------------------
# 8. >>> FINAL REFACTOR
#    One attention head
# ---------------------------------------------------------

class Head(nn.Module):

    def __init__(self):
        super().__init__()

        # Learnable Query projection.
        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        # Learnable Key projection.
        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        # Learnable Value projection.
        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        # Causal mask.
        #
        # A token can only look at itself
        # and tokens before it.
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )


    def forward(self, x):

        # x:
        # [batch, time, embedding]
        #
        # Example:
        # [4, 8, 16]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Q:
        # [4, 8, 4]
        #
        # K:
        # [4, 8, 4]
        #
        # V:
        # [4, 8, 4]


        # -------------------------------------------------
        # Attention scores
        # -------------------------------------------------

        scores = Q @ K.transpose(
            -2,
            -1
        )

        # [4, 8, 8]


        # Scale.
        scores = (
            scores
            / (K.shape[-1] ** 0.5)
        )


        # -------------------------------------------------
        # Causal mask
        # -------------------------------------------------

        T = x.shape[1]

        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )


        # -------------------------------------------------
        # Attention weights
        # -------------------------------------------------

        weights = F.softmax(
            scores,
            dim=-1
        )


        # -------------------------------------------------
        # Weighted V
        # -------------------------------------------------

        out = weights @ V

        # [4, 8, 4]

        return out


# ---------------------------------------------------------
# 9. >>> FINAL REFACTOR
#    Multi-head attention
# ---------------------------------------------------------

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        # Create several independent heads.
        self.heads = nn.ModuleList([
            Head()
            for _ in range(num_heads)
        ])

        # Mix the outputs from all heads.
        self.projection = nn.Linear(
            n_embd,
            n_embd,
            bias=False
        )


    def forward(self, x):

        # Run every head independently.
        head_outputs = [
            head(x)
            for head in self.heads
        ]

        # Concatenate:
        #
        # [4, 8, 4]
        # [4, 8, 4]
        # [4, 8, 4]
        # [4, 8, 4]
        #
        # becomes:
        #
        # [4, 8, 16]

        out = torch.cat(
            head_outputs,
            dim=-1
        )

        # Mix the heads.
        out = self.projection(out)

        return out


# ---------------------------------------------------------
# 10. >>> FINAL REFACTOR
#     Feed-forward network
# ---------------------------------------------------------

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Expand.
            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            # Non-linearity.
            nn.ReLU(),

            # Compress.
            nn.Linear(
                4 * n_embd,
                n_embd
            )
        )


    def forward(self, x):

        return self.network(x)


# ---------------------------------------------------------
# 11. >>> FINAL REFACTOR
#     Transformer block
# ---------------------------------------------------------

class Block(nn.Module):

    def __init__(self):
        super().__init__()

        # Normalize before attention.
        self.ln1 = nn.LayerNorm(
            n_embd
        )

        # Attention.
        self.sa = MultiHeadAttention()

        # Normalize before FFN.
        self.ln2 = nn.LayerNorm(
            n_embd
        )

        # Feed-forward network.
        self.ffwd = FeedForward()


    def forward(self, x):

        # -------------------------------------------------
        # Attention + residual
        # -------------------------------------------------

        x = (
            x
            + self.sa(
                self.ln1(x)
            )
        )


        # -------------------------------------------------
        # FFN + residual
        # -------------------------------------------------

        x = (
            x
            + self.ffwd(
                self.ln2(x)
            )
        )


        return x


# ---------------------------------------------------------
# 12. >>> MINI-GPT
# ---------------------------------------------------------

class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        # Token embeddings
        # -------------------------------------------------

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )


        # -------------------------------------------------
        # Position embeddings
        # -------------------------------------------------

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )


        # -------------------------------------------------
        # Transformer blocks
        # -------------------------------------------------

        self.blocks = nn.Sequential(
            *[
                Block()
                for _ in range(n_layer)
            ]
        )


        # -------------------------------------------------
        # Final LayerNorm
        # -------------------------------------------------

        self.ln_f = nn.LayerNorm(
            n_embd
        )


        # -------------------------------------------------
        # Language-model head
        # -------------------------------------------------

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )


    def forward(self, idx, targets=None):

        B, T = idx.shape

        # -------------------------------------------------
        # Token embeddings
        # -------------------------------------------------

        token_embeddings = (
            self.token_embedding_table(idx)
        )

        # [B, T, 16]


        # -------------------------------------------------
        # Position embeddings
        # -------------------------------------------------

        position_embeddings = (
            self.position_embedding_table(
                torch.arange(
                    T,
                    device=idx.device
                )
            )
        )

        # [T, 16]


        # -------------------------------------------------
        # Combine token + position
        # -------------------------------------------------

        x = (
            token_embeddings
            + position_embeddings
        )

        # [B, T, 16]


        # -------------------------------------------------
        # Transformer blocks
        # -------------------------------------------------

        x = self.blocks(x)


        # -------------------------------------------------
        # Final normalization
        # -------------------------------------------------

        x = self.ln_f(x)


        # -------------------------------------------------
        # Convert to logits
        # -------------------------------------------------

        logits = self.lm_head(x)

        # [B, T, vocab_size]


        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )


        return logits, loss


    # -----------------------------------------------------
    # Text generation
    # -----------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):

        self.eval()

        for _ in range(max_new_tokens):

            # Keep context within the model's
            # maximum context window.
            idx_cond = idx[:, -block_size:]


            # Run model.
            logits, _ = self(idx_cond)


            # Only use the final position.
            logits = logits[:, -1, :]


            # -------------------------------------------------
            # Temperature
            # -------------------------------------------------

            if temperature <= 0:
                raise ValueError(
                    "temperature must be greater than 0"
                )

            logits = (
                logits
                / temperature
            )


            # -------------------------------------------------
            # Top-k
            # -------------------------------------------------

            if top_k is not None:

                top_k = min(
                    top_k,
                    logits.shape[-1]
                )

                values, _ = torch.topk(
                    logits,
                    top_k
                )

                cutoff = values[:, [-1]]

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf")
                )


            # -------------------------------------------------
            # Convert logits to probabilities
            # -------------------------------------------------

            probabilities = F.softmax(
                logits,
                dim=-1
            )


            # -------------------------------------------------
            # Top-p
            # -------------------------------------------------

            if top_p is not None:

                sorted_probs, sorted_indices = (
                    torch.sort(
                        probabilities,
                        descending=True,
                        dim=-1
                    )
                )

                cumulative_probs = (
                    torch.cumsum(
                        sorted_probs,
                        dim=-1
                    )
                )

                remove_mask = (
                    cumulative_probs > top_p
                )

                # Keep the first token above
                # the threshold.
                remove_mask[:, 1:] = (
                    remove_mask[:, :-1]
                    .clone()
                )

                remove_mask[:, 0] = False

                sorted_probs = sorted_probs.masked_fill(
                    remove_mask,
                    0.0
                )

                # Renormalize.
                sorted_probs = (
                    sorted_probs
                    / sorted_probs.sum(
                        dim=-1,
                        keepdim=True
                    )
                )

                probabilities = torch.zeros_like(
                    probabilities
                ).scatter(
                    1,
                    sorted_indices,
                    sorted_probs
                )


            # -------------------------------------------------
            # Sample next token
            # -------------------------------------------------

            next_id = torch.multinomial(
                probabilities,
                num_samples=1
            )


            # Append token.
            idx = torch.cat(
                (idx, next_id),
                dim=1
            )


        return idx


# ---------------------------------------------------------
# 13. Create model
# ---------------------------------------------------------

model = MiniGPT()

print("\nModel:")
print(model)


# ---------------------------------------------------------
# 14. Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ---------------------------------------------------------
# 15. Train
# ---------------------------------------------------------

model.train()

for step in range(max_iters):

    X, Y = get_batch()

    logits, loss = model(
        X,
        Y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    if step % 20 == 0:

        print(
            "step:",
            step,
            "loss:",
            loss.item()
        )


# ---------------------------------------------------------
# 16. Generate
# ---------------------------------------------------------

print("\nGenerated text:")

start = torch.tensor(
    [[stoi["c"]]],
    dtype=torch.long
)

generated = model.generate(
    start,
    max_new_tokens=30,
    temperature=0.8,
    top_k=5
)

print(
    decode(
        generated[0].tolist()
    )
)