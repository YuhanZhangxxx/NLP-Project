# Rap Technique Detection Test Tool – Usage Guide

## Overview

`skill_detection/test/test_rap_techniques.py` is a unified test script with two separate interfaces:

1. **Interactive mode** – for manual testing and debugging  
2. **Dataset test** – for batch evaluation (accuracy, recall, F1, etc.)

## Interfaces

### 1. Interactive mode (default)

**How to run:**
```bash
# Default: interactive mode
python skill_detection/test/test_rap_techniques.py

# Explicit interactive
python skill_detection/test/test_rap_techniques.py --interactive
python skill_detection/test/test_rap_techniques.py -i
```

**Features:**
- **Single test**: type a line of text and press Enter
- **Batch test**: enter multiple lines starting with a number (e.g. `1. text`); script enters batch mode
- **Quit**: type `quit` or `exit`

**Example:**
```
Enter text (or first line for batch): After that round I pulled the clip titled it Where Your Voice Went and dropped the documentary
[results shown]

Enter text (or first line for batch): 1. Your first test text
[Batch mode] Enter more lines; empty line or 'end' to finish:
2. Your second test text
3. Your third test text
[empty line or 'end']
[batch results]
```

### 2. Dataset test

**How to run:**
```bash
python skill_detection/test/test_rap_techniques.py --test-dataset
python skill_detection/test/test_rap_techniques.py --dataset
python skill_detection/test/test_rap_techniques.py -d
```

**Behavior:**
- Loads `skill_detection/data/full_court_shot_correct.txt` and `skill_detection/data/full_court_shot_incorrect.txt`
- Runs all positive and negative examples
- Prints accuracy, recall, F1, confusion matrix, score distribution

**Example output:**
```
======================================================================
Full-Court Shot dataset test
======================================================================

Correct examples: 30
Incorrect examples: 22
======================================================================

[Testing correct examples]
----------------------------------------------------------------------
[OK] #1 : 100.0% - After that round the whole room watched your confi...
...
[Testing incorrect examples]
----------------------------------------------------------------------
[OK] #1 :   0.0% - After that round the room felt some kind of vibe I...
[FAIL] #2 : 100.0% - Mid battle the energy was different I clipped the ...
...
```

### 3. Single test from command line

```bash
python skill_detection/test/test_rap_techniques.py "Your text here"
```

### 4. Batch test from file

**Windows PowerShell:**
```powershell
Get-Content skill_detection/data/full_court_shot_correct.txt | python skill_detection/test/test_rap_techniques.py
```

**Linux/Mac:**
```bash
python skill_detection/test/test_rap_techniques.py < skill_detection/data/full_court_shot_correct.txt
```

## Interface independence

- **`test_dataset()`** – dataset test; loads data files and prints metrics  
- **`run_interactive_mode()`** – interactive; single and batch input  
- **`main()`** – entry point; chooses interface from CLI; default is interactive  

## Command-line options

| Option | Description |
|--------|-------------|
| (none) | Interactive mode (default) |
| `--interactive`, `-i` | Interactive mode |
| `--test-dataset`, `--dataset`, `-d` | Run dataset test |
| `--help`, `-h` | Help |
| `"text"` | Single test with given text |

## Code layout

```
skill_detection/test/test_rap_techniques.py
├── format_technique_score()   # Format score display
├── test_text()               # Single test
├── parse_batch_input()       # Parse batch input
├── batch_test()              # Batch test
├── test_dataset()            # Dataset test (standalone)
├── run_interactive_mode()     # Interactive mode (standalone)
└── main()                     # Entry point
```

## FAQ

**Q: How do I run the dataset test?**  
A: `python skill_detection/test/test_rap_techniques.py --test-dataset`

**Q: How do I use interactive mode?**  
A: Run `python skill_detection/test/test_rap_techniques.py` or add `--interactive`.

**Q: Can both interfaces run at once?**  
A: No; one interface per run, selected by arguments.

**Q: Does the dataset test modify data files?**  
A: No; it only reads the data files.
