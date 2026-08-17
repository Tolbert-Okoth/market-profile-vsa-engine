"""
Deep Literature Synthesizer: Tom Williams' Volume Spread Analysis (VSA) & Wyckoff Mechanics.
Produces comprehensive data/vsa_literature_synthesis.md from data/extracted_vsa_corpus.json.
"""
import json
import os

with open("data/extracted_vsa_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

mtm = corpus["mtm_251058.pdf"]
ch1_3 = corpus["1st3Chapters_5_24.pdf"]
beg = corpus["Beginners_guide.pdf"]

synthesis_content = """# Exhaustive Literature Synthesis: Volume Spread Analysis (VSA) & Wyckoff Theory

## Primary Literature Analyzed & Deeply Ingested
1. **Williams, Tom (2005/2011)**. *Master the Markets: Taking the Mystery out of the Financial Markets with Volume Spread Analysis*. TradeGuider Systems International. (185 pages) [`mtm_251058.pdf`](file:///c:/Users/Tolbert%20Okoth/scalper/data/mtm_251058.pdf)
2. **Williams, Tom & TradeGuider Team**. *Foundational Principles of Professional Market Manipulation*. (37 pages) [`1st3Chapters_5_24.pdf`](file:///c:/Users/Tolbert%20Okoth/scalper/data/1st3Chapters_5_24.pdf)
3. **TradeGuider Systems**. *The Beginner's Guide to Volume Spread Analysis*. (33 pages) [`Beginners_guide.pdf`](file:///c:/Users/Tolbert%20Okoth/scalper/data/Beginners_guide.pdf)

---

## 1. The Core Foundations of Volume Spread Analysis (VSA)

Volume Spread Analysis (VSA) is a proprietary methodology pioneered by **Tom Williams** (a former syndicate trader with a Beverly Hills stock trading group in the 1960s–70s), built upon the foundational market mechanics of **Richard D. Wyckoff**.

### The Core Premise of VSA
Financial markets are not driven by mathematical indicators or retail sentiment; they are driven by the imbalance between **Supply and Demand** created by the actions of **Professional Money ("Smart Money" / Institutional Syndicates)**:

$$\\text{Market Direction} = f(\\text{Price Spread}, \\text{Closing Position}, \\text{Volume Effort}, \\text{Background Context})$$

### The 3 Fundamental Wyckoff Laws
1. **The Law of Supply and Demand**:
   - When Demand exceeds Supply $\\implies$ Price must rise.
   - When Supply exceeds Demand $\\implies$ Price must fall.
   - *VSA Application*: Volume measures the activity of Smart Money; the spread shows the price movement created by that volume.
2. **The Law of Cause and Effect**:
   - An effect (a major price trend or breakout) must be preceded by a cause (a sustained period of Accumulation or Distribution).
   - The wider the cause (the horizontal trading range / Value Area), the larger the resulting trend effect.
3. **The Law of Effort vs. Result**:
   - **Effort** is represented by **Volume** (trading activity / contracts exchanged).
   - **Result** is represented by **Spread** (the price range achieved between High and Low).
   - *Harmony*: High Volume + Wide Spread in trend direction $\\implies$ Genuine initiative move.
   - *Divergence (Anomaly)*: High Volume + Narrow Spread $\\implies$ Absorption / Hidden opposition by Smart Money (Imminent Reversal!).
   - *Divergence (Anomaly)*: Low Volume + Wide Spread $\\implies$ Professional withdrawal (Fake move / Lack of interest!).

---

## 2. Quantitative Variable Dimensions on Every Bar

For every M5 bar $t$, VSA evaluates 4 continuous mathematical dimensions:

### A. Price Spread (Range)
$$\\text{Spread}_t = \\text{High}_t - \\text{Low}_t$$
- **Relative Spread**: $\\text{Spread\_Ratio}_t = \\frac{\\text{Spread}_t}{\\text{SMA}_{20}(\\text{Spread})}$
  - *Ultra-Wide Spread*: $\\text{Spread\_Ratio}_t \\ge 1.8$
  - *Wide Spread*: $1.3 \\le \\text{Spread\_Ratio}_t < 1.8$
  - *Average Spread*: $0.8 \\le \\text{Spread\_Ratio}_t < 1.3$
  - *Narrow Spread*: $0.5 \\le \\text{Spread\_Ratio}_t < 0.8$
  - *Ultra-Narrow Spread*: $\\text{Spread\_Ratio}_t < 0.5$

### B. Close Location within the Bar
$$\\text{Close\_Position}_t = \\frac{\\text{Close}_t - \\text{Low}_t}{\\text{High}_t - \\text{Low}_t} \\in [0.0, 1.0]$$
- **Upper Third**: $\\text{Close\_Position}_t \\ge 0.67$ (Dominant buyer closing control).
- **Middle Third**: $0.33 < \\text{Close\_Position}_t < 0.67$ (Two-sided equilibrium / battle).
- **Lower Third**: $\\text{Close\_Position}_t \\le 0.33$ (Dominant seller closing control).

### C. Volume (Effort)
$$\\text{Volume\_Ratio}_t = \\frac{\\text{Volume}_t}{\\text{SMA}_{20}(\\text{Volume})}$$
- **Ultra-High Volume**: $\\text{Volume\_Ratio}_t \\ge 2.0$ (Heavy Smart Money activity, climaxes, or absorption).
- **High Volume**: $1.3 \\le \\text{Volume\_Ratio}_t < 2.0$.
- **Average Volume**: $0.8 \\le \\text{Volume\_Ratio}_t < 1.3$.
- **Low / Drying Volume**: $\\text{Volume\_Ratio}_t \\le 0.70$ (Lack of professional participation / Tests).

### D. Background Context (Preceding Market State)
VSA dictates that no bar can be interpreted in isolation:
$$\\text{Background} = \\{\\text{Trend Direction}, \\text{Prior Support/Resistance}, \\text{Climactic Events in Last 20-50 Bars}\\}$$

---

## 3. Comprehensive Taxonomy of Signs of Strength (SoS)

Signs of Strength appear when Professional Money is **buying and absorbing floating supply**:

| SoS Pattern | Mathematical & Anatomical Conditions | Theoretical Interpretation (Tom Williams) |
| :--- | :--- | :--- |
| **SoS-1: Selling Climax** | $\\text{Down Bar} \\land \\text{Spread\_Ratio} \\ge 1.8 \\land \\text{Volume\_Ratio} \\ge 2.2 \\land \\text{Close\_Position} \\ge 0.35$ | Retail traders are dumping in panic; Smart Money steps in and buys up all available supply. The market cannot fall further because floating supply has been transferred to strong hands. |
| **SoS-2: Stopping Volume** | $\\text{Down Bar into Support} \\land \\text{Volume\_Ratio} \\ge 1.6 \\land \\text{Close\_Position} \\ge 0.40 \\land \\text{Spread} < \\text{Prior Spread}$ | High volume on a narrowing spread into support proves that professionals are placing limit buy orders that "stop" the downward momentum. |
| **SoS-3: Shakeout** | $\\text{Down Bar dipping below Support} \\land \\text{Volume\_Ratio} \\ge 1.5 \\land \\text{Close\_Position} \\ge 0.67$ | A sudden, deliberate violent down-spike below known support designed to trigger retail stop-losses and shake out weak holders before a markup. |
| **SoS-4: No Supply Test** | $\\text{Down/Narrow Bar} \\land \\text{Volume}_t < \\min(\\text{Vol}_{t-1}, \\text{Vol}_{t-2}) \\land \\text{Volume\_Ratio} \\le 0.70 \\land \\text{Close\_Position} \\ge 0.50$ | Smart Money dips price down to test if sellers are still present. Low volume proves floating supply has dried up. Market is primed to rally immediately! |
| **SoS-5: Testing in a Rising Market** | $\\text{Minor Pullback Bar} \\land \\text{Volume\_Ratio} \\le 0.65 \\land \\text{Close\_Position} \\ge 0.50 \\land P > \\text{EMA}_{24\\text{h}}$ | During an uptrend, a brief shallow dip on ultra-low volume confirms that no professional selling is present, clearing the path for upside continuation. |
| **SoS-6: Reverse Upthrust** | $\\text{Wide Spread Down} \\land \\text{Probe below Low} \\land \\text{Close\_Position} \\ge 0.75 \\land \\text{Lower Wick} \\ge 0.40$ | Price sweeps below support but closes on the absolute high of the bar, trapping breakout short sellers. |
| **SoS-7: Absorption at Resistance** | $\\text{Up Bar pushing through Resistance} \\land \\text{Volume\_Ratio} \\ge 1.5 \\land \\text{Close\_Position} \\ge 0.70 \\land \\text{Next Bar Holds}$ | Smart Money aggressively absorbs all overhead limit sell orders to cleanly break out through a resistance/VAH level. |

---

## 4. Comprehensive Taxonomy of Signs of Weakness (SoW)

Signs of Weakness appear when Professional Money is **selling and distributing inventory to uninformed buyers**:

| SoW Pattern | Mathematical & Anatomical Conditions | Theoretical Interpretation (Tom Williams) |
| :--- | :--- | :--- |
| **SoW-1: Buying Climax** | $\\text{Up Bar} \\land \\text{Spread\_Ratio} \\ge 1.8 \\land \\text{Volume\_Ratio} \\ge 2.2 \\land \\text{Close\_Position} \\le 0.65$ | Retail traders rush in with FOMO on good news; Smart Money unloads their accumulated inventory into the retail buy orders, capping the market. |
| **SoW-2: Upthrust (UT)** | $\\text{Up Bar probing above Resistance} \\land \\text{Upper Wick} \\ge 0.40 \\land \\text{Close\_Position} \\le 0.30 \\land (\\text{High Vol} \\lor \\text{Low Vol})$ | Price probes above resistance to trigger buy-stops and entice breakout longs, but immediately closes on the lows. Traps buyers at the top of the market. |
| **SoW-3: Pseudo Upthrust** | $\\text{Up Bar} \\land \\text{Volume\_Ratio} \\le 0.70 \\land \\text{Close\_Position} \\le 0.35$ | An up move on drying volume that fails to close near the highs, indicating a complete lack of professional sponsorship. |
| **SoW-4: No Demand Bar** | $\\text{Up/Narrow Bar} \\land \\text{Volume}_t < \\min(\\text{Vol}_{t-1}, \\text{Vol}_{t-2}) \\land \\text{Volume\_Ratio} \\le 0.70 \\land \\text{Close\_Position} \\le 0.50$ | Price closes up on ultra-low volume. Proves that Smart Money has zero interest in higher prices. High statistical precursor to a sharp sell-off! |
| **SoW-5: End of a Rising Market** | $\\text{Narrow Spread Up Bar} \\land \\text{Volume\_Ratio} \\ge 2.0 \\land \\text{Close\_Position} \\le 0.60$ | Enormous volume effort with almost zero upward progress (narrow spread). Proves massive hidden institutional selling into the buying volume. |
| **SoW-6: Top Reversal / Squat** | $\\text{Wide Spread Down following Up Move} \\land \\text{Volume\_Ratio} \\ge 1.5 \\land \\text{Close\_Position} \\le 0.25$ | Aggressive institutional liquidation overwhelming all residual buy orders. |

---

## 5. Synergistic Fusion: VSA + Dalton Auction Market Theory

Integrating Tom Williams' VSA with James Dalton's Market Profile produces an institutional synergy that resolves the blind entry problem:

```mermaid
graph TD
    A["Price at Dalton Reference Level"] --> B{"Location in Auction Profile?"}
    
    B -->|At Prior VAL / Discount| C["Look for VSA Sign of Strength (SoS)"]
    C --> C1["SoS-4: No Supply Test (Low Volume)"]
    C --> C2["SoS-2: Stopping Volume (High Vol Absorption)"]
    C --> C3["SoS-3: Shakeout / Reverse Upthrust"]
    C1 & C2 & C3 --> C4["CONFIRMED BUY: Rotate to POC & VAH"]
    
    B -->|At Prior VAH / Premium| D["Look for VSA Sign of Weakness (SoW)"]
    D --> D1["SoW-4: No Demand Bar (Low Volume)"]
    D --> D2["SoW-2: Upthrust (Trapped Longs)"]
    D --> D3["SoW-5: End of Rising Market (Hidden Selling)"]
    D1 & D2 & D3 --> D4["CONFIRMED SELL: Rotate to POC & VAL"]
    
    B -->|At Prior POC / Fair Value| E["Evaluate Effort vs Result"]
    E -->|High Volume + Wide Spread| E1["Acceptance: Hold Runner"]
    E -->|High Volume + Narrow Spread / Upthrust| E2["Reversal: Take Profit TP1"]
```

### 1. Value Area Low (VAL) Long Confirmations:
- A Dalton 80% Rule Long entry from $\\text{VAL}$ is only triggered if accompanied by:
  1. A **No Supply Test** (volume $< 0.7\\times\\text{SMA}_{20}$ on downward probe).
  2. **Stopping Volume** or **Selling Climax Absorption** (volume $\\ge 1.6\\times\\text{SMA}_{20}$ with close in upper half).
  3. A **Shakeout / Lower Rejection Wick** ($\ge 35\\%$ lower wick).

### 2. Value Area High (VAH) Short Confirmations:
- A Dalton 80% Rule Short entry from $\\text{VAH}$ is only triggered if accompanied by:
  1. A **No Demand Bar** (up candle on drying volume $< 0.7\\times\\text{SMA}_{20}$).
  2. An **Upthrust** (upper wick $\ge 35\\%$ with close in lower third).
  3. **Buying Climax Exhaustion** (ultra-high volume failing to advance).

### 3. Point of Control (POC) Management:
- When price rotates into the $\text{POC}$, evaluate the **Effort vs Result** of the arrival candle:
  - If a **No Demand** or **Upthrust** appears at $\text{POC} \implies \textbf{Take Profit 1 Immediately}$ (reversal expected).
  - If heavy absorption volume breaks cleanly through $\text{POC} \implies \textbf{Trail to VAH/VAL}$.
"""

with open("data/vsa_literature_synthesis.md", "w", encoding="utf-8") as f:
    f.write(synthesis_content)

print(f"Comprehensive VSA Literature Synthesis written to data/vsa_literature_synthesis.md ({len(synthesis_content):,} chars)")
