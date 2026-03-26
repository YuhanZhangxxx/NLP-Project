# Full-Court Shot – Grammar and Features

## Overview

Full-Court Shot in English is a “three-beat combo” pattern: grammatically regular, like reporting a sequence of events. This doc describes the grammar-based detection used in the code.

## Core grammar

### 1. Triple-verb pattern (strongest signal)

**Skeleton:**
```
I + VERB1 + OBJECT1 + VERB2 + OBJECT2 + VERB3 + OBJECT3
```

**Verb roles:**
- **VERB1 (capture/create)**: `pulled`, `clipped`, `grabbed`, `captured`, `caught`, `snapped`, `recorded`, `ran`, `printed`, `wrote`, `filed`, `issued`, `opened`, `logged`, `tracked`, `made`, `created`, `pushed`, `posted`, `built`, `constructed`, `assembled`, `erected`
- **VERB2 (naming)**: `titled`, `named`, `called`, `labeled`, `tagged` + `it/this/that/the`
- **VERB3 (release)**: `dropped`, `released`, `unveiled`, `debuted`, `aired`, `broadcast`, `streamed`, `premiered`, `uploaded`, `published`, `opened`, `launched`, `presented`

**Properties:**
- Same subject `I`
- Three past transitive verbs in a row
- Naming verb in the middle (`titled` / `named` / `called`)
- “I did three steps” rhythm

**Examples:**
```
I pulled the replay titled it Where His Heart Went and aired the documentary
I clipped the moment named it Live Disappearance and released the case file
I built the museum display titled it The Fall Of The Front and opened the exhibit
```

### 2. Two-part structure

**Template:**
```
[time/condition] + [opponent collapse] + I + [triple action chain]
```

**Part 1 (opponent state):**
- Time: `By the end of the round`, `The moment`, `After that round`, `Mid battle`
- Collapse: `your X was Y`, `your X vanished/disappeared`, `your X got stolen/deleted`

**Part 2 (production chain):**
- `I + VERB1 + OBJ1 + VERB2 + OBJ2 + VERB3 + OBJ3`

### 3. Connectors

- `and and and` – repeated “and”, flow
- `and then` – sequence
- `so I` – cause
- `and I` – simple link

### 4. Time-frame openers

- `By the end of`, `The moment`, `As soon as`, `After that round`, `Mid battle`, `Soon as`, `The second`, `When the`, `That round`, `One beat switch later`

### 5. Title patterns

- `Where His X Went` – “where X went”
- `The Day He Y` – “the day he Y”
- `Live Noun`, `Missing Since Noun`, `X Autopsy`, `X Heist`, `The Fall Of X`, `Exhibit A`, `The Exact Frame`

### 6. Person configuration

- `you/your` – defeated (victim)
- `I` – director/reporter/judge (producer)

## Detection in code

In `detect_full_court_shot()` the following grammar checks are used:

1. **Triple-verb pattern** (+0.3): `I + VERB1 + ... + VERB2 (titled/named/called) + ... + VERB3`, with optional connectors.
2. **Time-frame opener** (+0.1): time phrase at start of sentence.
3. **Two-part structure** (+0.15): time + opponent state + I + actions.
4. **Connector preference** (+0.05–0.1): `and and and`, `and then`, `so I`, `and I`.
5. **Person config** (+0.1): both `you/your` and `I` present.
6. **Title grammar** (+0.08–0.15): title-like noun phrase or clause.

## Effect

With these grammar features:
- **Correct-example average score**: 68.5% → 90.8% (+22.3%)
- **Many correct examples hit 100%**
- **Accuracy remains 97.5%**
- **Incorrect-example average**: 0.8% (low, no extra false positives)

## Strongest grammar signal

The main pattern we detect is: same subject `I`, three past transitive verbs in a row, with a naming verb (`titled` / `named` / `called`) in the middle. That is the core Full-Court Shot grammar.
