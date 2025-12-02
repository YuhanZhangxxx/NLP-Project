# Overfitting Analysis - First 20 Lines Test

## Summary

Tested first 20 lines of 12 songs from `data/lyrics/` directory using v2_plus model.

## Results

- **Total songs tested**: 12
- **Songs with known labels**: 9
- **Correct predictions**: 1/9 (11.1%)
- **Average p_good**: 0.463

## Performance by Split

| Split | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Train | 0/8 | 8 | 0.0% |
| Test  | 0/1 | 1 | 0.0% |

## Detailed Results

| File | Split | Prediction | p_good | True Label | Correct |
|------|-------|------------|--------|------------|---------|
| A 100 Clip.txt | train | **good** | 0.718 | good | ✓ |
| AM to PM.txt | train | bad | 0.448 | good | ✗ |
| B-Boy Stance Dirty.txt | train | bad | 0.228 | good | ✗ |
| Bellybutton.txt | train | bad | 0.499 | good | ✗ |
| Blood Pressure Dirty.txt | train | bad | 0.398 | good | ✗ |
| Cassidy - Get No Better... | test | bad | 0.459 | good | ✗ |
| Cassidy - Shaq Kobe... | train | bad | 0.465 | good | ✗ |
| Cassidy- Respectfully.txt | train | bad | 0.369 | good | ✗ |
| Fabolous - Affirmative Action.txt | train | bad | 0.386 | good | ✗ |

## Key Findings

### 1. **Severe Overfitting Indication**
- Model trained on full songs but tested on first 20 lines only
- **0% accuracy on training set songs** when using partial text
- This suggests the model memorized full-song patterns rather than learning general quality indicators

### 2. **Data Quality Issues**
- First 20 lines often contain:
  - Transcription artifacts ("We'll be right back", "Do not censor")
  - Ad content
  - Incomplete thoughts/sentences
- These artifacts may not represent actual song quality

### 3. **Model Behavior**
- Most predictions are **BAD** (8/9) even though all true labels are **GOOD**
- Average p_good (0.463) is below threshold (0.5)
- Model seems to require full context to make accurate predictions

## Recommendations

1. **Test on full songs** to see if accuracy improves
2. **Improve cleaning** to better remove transcription artifacts
3. **Consider segment-level training** if partial text prediction is needed
4. **Cross-validation** on different song segments to check generalization
5. **Feature analysis** to understand what the model is actually learning

## Conclusion

The extremely low accuracy (11.1%) on first 20 lines, especially for training set songs, strongly suggests **overfitting**. The model appears to have memorized full-song patterns rather than learning generalizable quality indicators that work on partial text.

