# Ask It Anything (About Tennis)

This app answers questions about tennis. It's honest about what it doesn't know —
instead of guessing, it tells you when it can't find a good answer.

---

## Starting the app

You'll need to do this once in a terminal:

```
streamlit run app.py
```

Then open the link it gives you (usually `http://localhost:8501`) in your browser.

---

## Asking a question

Type your question in the box and press Enter.

**Good questions to try:**
- How does a tiebreak work?
- How many Grand Slam titles has Rafael Nadal won?
- What surface is the French Open played on?
- Who is Billie Jean King?
- What is a volley?

The app only knows about tennis — it won't answer questions about other topics,
and it will say so.

---

## Reading the answer

### The coloured badge

Every response starts with one of three decisions:

| Badge | What it means |
|-------|---------------|
| 🟢 **ANSWER** | Found good evidence. The answer is reliable. |
| 🟡 **SEARCH AGAIN → answered** | The first search wasn't enough, so it looked harder and found something. Treat this answer a little more cautiously. |
| 🔴 **ABSTAIN** | Not enough evidence to answer confidently. The app is refusing rather than guessing. |

### The answer text

For 🟢 and 🟡, you get the answer with inline citations like `[1]` or `[2]`.
Those numbers match the source passages shown below.

For 🔴 ABSTAIN, you get a short explanation of why it couldn't answer — for
example: "the passages don't support this" or "nothing relevant in the corpus."

### Sources & signals (the expandable section)

Click **Sources & signals** to see:
- Which Wikipedia passages it retrieved (and how relevant each one was)
- Which passages it actually cited in the answer
- The similarity score — higher means more relevant (above 0.30 is the threshold)

---

## Choosing a pipeline

At the top of the page, pick between:

- **Abstaining RAG** — the honest one. Can say "I don't know."
- **Basic RAG (baseline)** — always gives an answer, even if the evidence is weak.

Try the same question on both and compare. On questions about players or events
not in the corpus (like Steffi Graf or Pete Sampras), the abstaining version will
refuse while the basic version will attempt an answer it can't really support.

---

## Why did it abstain on my question?

A few common reasons:

- **The topic isn't in the corpus.** The app knows about ~24 tennis players and
  tournaments. If you ask about someone not in that list (e.g. Pete Sampras,
  Steffi Graf), it correctly says it doesn't know.
- **The question is off-topic.** Anything not about tennis will score low on
  retrieval and get refused immediately.
- **The evidence is thin.** Sometimes the relevant passages exist but don't
  contain enough detail to support a confident answer.

This is the point of the app — it's better to say "I don't know" than to make
something up.

---

## What the app knows about

The corpus covers approximately 24 Wikipedia articles:

**Players:** Andre Agassi, Billie Jean King, Martina Navratilova, Novak Djokovic,
Rafael Nadal, Roger Federer, Serena Williams

**Grand Slams:** Australian Open, French Open, US Open, Wimbledon

**Concepts:** Tennis scoring, tiebreaks, serves, volleys, clay/grass/hard courts,
tennis balls, tennis rackets, tennis courts, Grand Slam tennis, Fed Cup

**Not in the corpus** (so it will abstain on questions about):
Pete Sampras, Steffi Graf, ATP Tour, WTA Tour, Davis Cup

---

## Tips

- Short, factual questions work best.
- If you get ABSTAIN and want to dig deeper, try rephrasing or asking a more
  specific sub-question.
- The sources panel tells you exactly what the app was working from — if the
  relevant passage is there but it still abstained, the support check decided
  the answer wasn't grounded enough.
