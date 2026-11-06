# Follow-up protocol for input screening

The original 81 fits showed a larger mean loss for the gated MLP than for the original MLP. This follow-up evaluates a different intervention on split seeds 3, 4, and 5. These splits use the same three datasets and can share rows with the earlier splits. They are fresh split assignments, not external validation datasets.

## Fixed method

Fit a feature selector on the 80% optimization portion of the outer training rows, using the same stratified split as the MLP's checkpoint validation. For each numerical column, estimate mutual information with the target using at most 16 quantile bins. Shuffle the optimization labels 64 times. Record the maximum feature score after each shuffle, and compare each observed score to that null distribution with a plus-one permutation count. Keep columns with a permutation score at or below 0.05. If the mask would be empty, keep all columns. Apply the same learned mask to validation and test data. The method never uses the identities of the injected noise columns when making selections.

Train the upstream MLP after screening. The two MLPs share the same architecture settings, optimizer, checkpoint rule, 300-epoch cap, data splits, and paired noise draws. Test 0%, 50%, and 100% added columns on MagicTelescope, pol, and house_16H. Fit both models on the same Mac and Python environment. Log the number of kept original and injected columns for analysis after selection.

## Hypothesis and decision rule

**H3.** At 100% added columns, screening increases mean held-out accuracy by at least 1.00 percentage point over the original MLP. At 0% added columns, mean accuracy decreases by no more than 1.00 percentage point. Both conditions must hold to call the hypothesis supported.

Average the nine paired screened-minus-original differences at each noise level. Report the clean and noisy accuracy, the count of improved pairs, and dataset-level gains. The 50% noise comparison is descriptive. Accuracy and column retention are the main outcomes. Timing is diagnostic. Retain the earlier gate result separately. Do not revise the selector thresholds based on the test scores in this run.
