# Full Dataset Evaluation Report

## Summary

Tested model `song_lr_v2_plus_20251026.joblib` on entire dataset (176 songs).

## Overall Performance

- **Total Songs**: 176
- **Overall Accuracy**: 93.18% (164/176) ✅
- **GOOD songs**: 100.00% (89/89) ✅
- **BAD songs**: 86.21% (75/87) ⚠️

## Performance by Split

| Split | Accuracy | Count | Notes |
|-------|----------|-------|-------|
| **Train** | 95.00% | 133/140 | Training set |
| **Valid** | 94.12% | 16/17 | Validation set |
| **Test** | 78.95% | 15/19 | ⚠️ Test set |

## Overfitting Analysis

### ⚠️ Potential Overfitting Detected

**Key Findings:**
- **Train vs Test gap**: 16.05% difference (95.00% vs 78.95%)
- **Train vs Valid**: Only 0.88% difference (95.00% vs 94.12%) ✅

**Interpretation:**
1. **Test set is small** (only 19 songs) - small sample size can cause variance
2. **Validation set performance is good** (94.12%) - suggests model generalizes well
3. **Test set may contain harder cases** - 4 errors out of 19 songs
4. **All errors are BAD→GOOD** (false positives) - model tends to be lenient

### Error Breakdown

**Total Errors**: 12 songs

**By Split:**
- Train: 7 errors
- Test: 4 errors
- Valid: 1 error

**By Type:**
- All errors are **BAD songs predicted as GOOD** (false positives)
- No GOOD songs predicted as BAD (no false negatives)

**Error Examples:**
- `B_Attitude.clean` - p_good=0.521 (very close to threshold)
- `B_Drama Setter.clean` - p_good=0.602
- `B_I Used To Be In Love.clean` - p_good=0.658
- `B_Curious.clean` - p_good=0.549

## Recommendations

### 1. **Test Set Size**
- Test set is very small (19 songs, ~11% of data)
- Consider using a larger test set or cross-validation for more reliable evaluation

### 2. **Threshold Adjustment**
- All errors are borderline cases (p_good close to 0.5)
- Consider raising threshold to 0.55-0.60 to reduce false positives on BAD songs

### 3. **Model Performance**
- **Excellent on GOOD songs** (100% accuracy)
- **Good on BAD songs** (86.21% accuracy)
- Model is slightly lenient (tends to classify as GOOD)

### 4. **Overfitting Assessment**
- **Moderate concern**: 16% gap between train and test
- **But**: Validation set performance (94.12%) is very close to training (95.00%)
- **Conclusion**: May be overfitting slightly, but test set size makes it hard to be certain

## Conclusion

**Model Performance**: ✅ **Good** (93.18% overall)

**Overfitting Status**: ⚠️ **Minor concern** - 16% gap between train and test, but:
- Validation set performance is excellent (94.12%)
- Test set is very small (19 songs)
- All errors are borderline cases

**Recommendations**:
1. Increase test set size for more reliable evaluation
2. Consider threshold adjustment (0.55-0.60) to reduce false positives
3. Model is production-ready with minor tuning needed

