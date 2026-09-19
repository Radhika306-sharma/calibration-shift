# Classifier Calibration Under Distribution Shift: When Do Confidence Scores Remain Probabilities?

**Author:** Radhika Sharma  
**Course:** Probability and Statistics

---

## Abstract

Predicted probabilities from machine-learning classifiers are often interpreted as confidence estimates, but that interpretation may fail when a model is evaluated on a population different from the one used for training. This paper examines classifier calibration under increasing distribution shift using the IBM Telco Customer Churn dataset. The study defines a base training population and five evaluation populations constructed along two business-relevant axes: contract type and customer tenure. The resulting evaluation settings combine changes in feature distributions with changes in the churn rate. Logistic Regression, Gaussian Naive Bayes, Random Forest, an RBF-kernel Support Vector Machine, and XGBoost are evaluated using accuracy, Brier score, and Expected Calibration Error (ECE). Paired bootstrap comparisons, paired \(t\)-tests, Wilcoxon signed-rank tests, and Pearson and Spearman correlations are used to assess differences between models and the association between distributional divergence and calibration error.

The results show substantial heterogeneity across classifiers. Logistic Regression has the lowest ECE at every shift level, with ECE values of 0.024, 0.156, 0.088, 0.038, and 0.031 from Shift 0 through Shift 4, respectively. In contrast, the SVM's accuracy decreases from 0.691 at Shift 0 to 0.031 at Shift 4, while its ECE increases from 0.055 to 0.478. Gaussian Naive Bayes also exhibits severe calibration deterioration, with ECE increasing from 0.290 to 0.638. Across all model–shift observations, PSI and ECE are positively associated: Pearson's \(r=0.512\) (\(p=0.0089\)) and Spearman's \(\rho=0.573\) (\(p=0.0028\)). However, correlations estimated separately for each model are not statistically significant, reflecting the small number of shift levels per model. These findings indicate that confidence scores do not retain a uniform probabilistic interpretation under distribution shift. In this experiment, calibration behavior depends strongly on the model and the shifted population, so in-distribution calibration alone is insufficient evidence that predicted probabilities will remain reliable after deployment.

---

## 1. Introduction

Predictive probabilities produced by machine-learning classifiers are widely used as decision-support signals. A model that assigns a probability of 0.90 to a positive prediction is commonly interpreted as making a probabilistic claim: among cases receiving predictions near 0.90, approximately 90% should belong to the positive class. This property is known as **calibration**. Calibration is distinct from accuracy: a model can classify many observations correctly while producing poorly calibrated probabilities, or it can produce comparatively reliable probabilities despite having lower classification accuracy.

Calibration is especially important when a classifier is deployed outside the population represented in its training data. In such settings, the joint distribution of features and outcomes may differ between training and deployment. The literature commonly distinguishes changes in the feature distribution, changes in class proportions, and changes in the relationship between features and labels (Quiñonero-Candela et al., 2009). These forms of dataset shift can affect not only predictive accuracy but also the interpretation of model scores as probabilities.

Previous research has shown that calibration is model-dependent even under conventional evaluation conditions. Niculescu-Mizil and Caruana (2005) found systematic differences in the probability estimates produced by several supervised-learning methods, including support-vector machines, Naive Bayes, and boosted trees. Guo et al. (2017) demonstrated that modern neural networks can be substantially miscalibrated and showed that post-hoc temperature scaling can improve calibration on held-out data. These results establish that a classifier's score should not automatically be treated as a well-calibrated probability.

Distribution shift introduces a further concern. Ovadia et al. (2019) evaluated predictive uncertainty under a range of dataset shifts and reported that uncertainty quality generally deteriorated as the shift increased, with performance under shift not reliably predicted by in-distribution performance. Their study focused primarily on deep-learning and uncertainty-estimation methods, whereas the present paper examines classical and widely used tabular classifiers in a single business dataset.

This paper asks the following question:

> **When a classifier's training and evaluation distributions diverge, does its predicted confidence remain a valid estimate of the probability of correctness, and how does this behavior differ across model families?**

The analysis uses the IBM Telco Customer Churn dataset and constructs five increasingly shifted evaluation populations using contract type and tenure as natural drift axes. The study evaluates five classifiers using accuracy, Brier score, and Expected Calibration Error (ECE). It also examines pairwise differences between models and the association between a distribution-shift measure and calibration error.

The paper does not claim that the selected shift design represents all forms of deployment shift. Rather, it provides an empirical examination of how five classifiers behave under one explicitly defined combined covariate and prior-probability shift.

---

## 2. Related Work

### 2.1 Calibration and probabilistic prediction

Calibration concerns the agreement between predicted probabilities and observed frequencies. For a binary classifier, a prediction of \(p\) is calibrated when the conditional frequency of the positive outcome among cases assigned probability \(p\) is also \(p\), subject to the usual estimation issues that arise because exact probability values may have limited or zero empirical support.

Reliability diagrams provide a visual comparison between predicted probabilities and empirical outcome frequencies. DeGroot and Fienberg (1983) established the importance of comparing probabilistic forecasts with observed frequencies, and reliability diagrams have remained a widely used diagnostic for probabilistic prediction.

The Brier score provides another evaluation measure. For binary outcomes, it is the mean squared difference between the predicted probability and the observed outcome. Because it is a squared-error score, it reflects both calibration and the concentration or sharpness of predictions. Lower values indicate better probabilistic predictions, although the Brier score should not be interpreted as a pure calibration measure.

Expected Calibration Error is a binned summary of the difference between average predicted probability and empirical frequency. It is useful as a compact diagnostic, but it is an estimate that depends on the binning scheme and sample size. Accordingly, ECE is most informative when considered alongside reliability diagrams and other probabilistic scores rather than as a complete definition of calibration.

### 2.2 Model-specific calibration behavior

Niculescu-Mizil and Caruana (2005) compared probability estimates produced by several supervised-learning algorithms and documented systematic differences in their calibration behavior. In particular, the probability estimates of maximum-margin methods can be concentrated away from the extremes, while other models may display different forms of distortion. Their analysis supports the view that the model family and training objective matter when predicted scores are interpreted probabilistically.

Guo et al. (2017) examined calibration in modern neural networks and reported that high classification accuracy does not guarantee well-calibrated confidence estimates. They also evaluated temperature scaling as a post-hoc calibration method. Although the present paper does not use neural networks or temperature scaling, this work motivates the broader distinction between predictive performance and confidence reliability.

Ng and Jordan (2002) compared generative and discriminative approaches, using Naive Bayes and Logistic Regression as representative examples. Their results concern classification behavior and asymptotic properties rather than calibration under distribution shift specifically. Nevertheless, the comparison provides useful background for interpreting differences between these model families in the present experiment. It does not, by itself, establish why one model must be better calibrated than another in the current dataset.

### 2.3 Calibration under distribution shift

Dataset shift describes the situation in which the data-generating distribution used at deployment differs from the training distribution. Quiñonero-Candela et al. (2009) distinguish several forms of shift, including covariate shift, prior-probability shift, and concept shift. The shift design in this paper changes both the feature composition of the evaluation populations and their churn rates. It is therefore treated as a combined covariate and prior-probability shift rather than as a pure instance of only one shift type.

Ovadia et al. (2019) conducted a large empirical investigation of predictive uncertainty under dataset shift. Their results showed that uncertainty quality generally declines as the shift becomes more severe and that good in-distribution calibration does not reliably imply good calibration under shifted conditions. The present study follows that evaluation perspective while differing in dataset type, model family, and experimental scope: it uses tabular customer data and five classical classifiers rather than a broad collection of deep-learning uncertainty methods.

The existing literature therefore motivates three considerations relevant to this study:

1. classifier scores can be miscalibrated even without distribution shift;
2. calibration and accuracy measure different aspects of predictive performance; and
3. in-distribution calibration does not guarantee calibration on shifted populations.

The present experiment examines these issues jointly in a controlled, domain-specific shift design.

---

## 3. Methodology

### 3.1 Calibration and Expected Calibration Error

For a binary classifier, let \(\hat p_i\) denote the predicted probability of the positive class for observation \(i\), and let \(y_i\in\{0,1\}\) denote the observed outcome. A classifier is ideally calibrated when the predicted probabilities agree with the corresponding conditional outcome frequencies.

Because finite samples contain relatively few observations at any exact probability value, predictions are grouped into probability bins. For bin \(B_k\), let \(n_k\) be the number of observations, let \(\operatorname{conf}(B_k)\) be the mean predicted probability, and let \(\operatorname{acc}(B_k)\) denote the empirical positive-class frequency. The binned Expected Calibration Error is

\[
\operatorname{ECE}
=
\sum_{k=1}^{K}
\frac{n_k}{n}
\left|
\operatorname{acc}(B_k)-\operatorname{conf}(B_k)
\right|.
\]

Lower ECE indicates closer agreement between the binned predicted probabilities and observed frequencies. ECE is an approximation and depends on the selected binning procedure; it should therefore be interpreted together with the reliability diagrams and Brier scores.

### 3.2 Brier Score

For binary outcomes, the Brier score is

\[
\operatorname{BS}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat p_i-y_i)^2.
\]

Lower values indicate better probabilistic predictions under this loss. The Brier score jointly reflects calibration and the informativeness or sharpness of the predictions, so it is not equivalent to ECE.

### 3.3 Distribution-shift measure

Shift magnitude was quantified using the Population Stability Index (PSI). PSI was calculated to provide an independent numerical measure of the divergence between the base population and each evaluation population. KL-divergence was also computed for the numeric features tenure, monthly charges, and total charges.

The reported PSI values increase monotonically from approximately 0.015 for Shift 0 to 8.42 for Shift 4. KL-divergence follows the same intended ordering on the evaluated numeric features. These measures were used to quantify the ordering of the constructed evaluation populations; they do not establish that the populations differ through only one mechanism.

### 3.4 Statistical testing

For each shift level, paired bootstrap resampling was used to generate sampling distributions for model-specific ECE. The same bootstrap draws were reused across models to preserve pairing between model comparisons.

The analysis included:

- paired \(t\)-tests between every pair of models at every shift level;
- Wilcoxon signed-rank tests as a non-parametric robustness check;
- Pearson correlation between PSI and ECE; and
- Spearman correlation between PSI and ECE.

The pairwise tests evaluate differences in the paired bootstrap ECE values. The correlation analyses evaluate the association between the numerical shift measure and ECE. They do not, by themselves, establish a causal relationship between shift magnitude and calibration error.

---

## 4. Experimental Setup

### 4.1 Dataset

The IBM Telco Customer Churn dataset contains 7,043 customers and a binary churn target. Two variables were selected as natural drift axes:

- **Contract type:** month-to-month, one-year, or two-year;
- **Tenure:** number of months as a customer, ranging from 0 to 72.

Both variables are associated with churn behavior in the dataset and provide a business-relevant basis for constructing distinct evaluation populations.

### 4.2 Shift design

The base population and five evaluation populations were defined as follows:

| Test set | Population |
|---|---|
| Base (train) | Month-to-month contracts, tenure 0–12 months |
| Shift 0 (in-distribution control) | Held-out month-to-month contracts, tenure 0–12 |
| Shift 1 (mild) | Month-to-month contracts, tenure 12–24 |
| Shift 2 (moderate) | One-year contracts, tenure 24–48 |
| Shift 3 (strong) | Two-year contracts, tenure 48–60 |
| Shift 4 (severe) | Two-year contracts, tenure 60–72 |

Moving from Shift 0 to Shift 4 changes both contract composition and tenure. The churn rate also falls as the evaluation populations move toward longer-tenure and more committed customers. The design therefore combines feature-distribution changes with changes in the class prior.

PSI and KL-divergence on tenure, monthly charges, and total charges increase monotonically from Shift 0 to Shift 4. PSI increases from approximately 0.015 at Shift 0 to 8.42 at Shift 4.

### 4.3 Models

Five classifiers were trained on the base population using a shared preprocessing pipeline:

1. Logistic Regression;
2. Gaussian Naive Bayes;
3. Random Forest;
4. Support Vector Machine with an RBF kernel and probability estimates enabled; and
5. XGBoost.

Categorical variables were one-hot encoded, and numerical variables were scaled. The preprocessing pipeline was fitted using the base training data only.

### 4.4 Evaluation

For each model and shift level, the analysis computed:

- classification accuracy;
- Brier score;
- Expected Calibration Error; and
- a reliability diagram comparing predicted probabilities with observed positive-class frequencies.

The same fixed base training distribution was used across evaluation populations. This design makes it possible to compare how the models' probabilities behave as the evaluation population moves away from the training population under the specified shift construction.

---

## 5. Results

### 5.1 Accuracy and calibration across shift levels

| Test set | Model | Accuracy | Brier Score | ECE |
|---|---|---:|---:|---:|
| Shift 0 (in-dist.) | Logistic Regression | 0.705 | 0.195 | **0.024** |
| Shift 0 | Naive Bayes | 0.674 | 0.298 | 0.290 |
| Shift 0 | Random Forest | 0.656 | 0.214 | 0.079 |
| Shift 0 | SVM (RBF) | 0.691 | 0.201 | 0.055 |
| Shift 0 | XGBoost | 0.676 | 0.221 | 0.104 |
| Shift 1 (mild) | Logistic Regression | 0.640 | 0.228 | 0.156 |
| Shift 1 | Naive Bayes | 0.560 | 0.403 | 0.405 |
| Shift 1 | Random Forest | 0.634 | 0.231 | 0.103 |
| Shift 1 | SVM (RBF) | 0.607 | 0.228 | **0.089** |
| Shift 1 | XGBoost | 0.609 | 0.277 | 0.200 |
| Shift 2 (moderate) | Logistic Regression | 0.894 | 0.100 | **0.088** |
| Shift 2 | Naive Bayes | 0.558 | 0.416 | 0.422 |
| Shift 2 | Random Forest | 0.697 | 0.200 | 0.293 |
| Shift 2 | SVM (RBF) | 0.259 | 0.250 | 0.395 |
| Shift 2 | XGBoost | 0.606 | 0.268 | 0.332 |
| Shift 3 (strong) | Logistic Regression | 0.960 | 0.039 | **0.038** |
| Shift 3 | Naive Bayes | 0.570 | 0.418 | 0.427 |
| Shift 3 | Random Forest | 0.805 | 0.160 | 0.323 |
| Shift 3 | SVM (RBF) | 0.040 | 0.259 | 0.470 |
| Shift 3 | XGBoost | 0.668 | 0.247 | 0.391 |
| Shift 4 (severe) | Logistic Regression | 0.969 | 0.031 | **0.031** |
| Shift 4 | Naive Bayes | 0.362 | 0.631 | 0.638 |
| Shift 4 | Random Forest | 0.656 | 0.230 | 0.418 |
| Shift 4 | SVM (RBF) | 0.031 | 0.259 | 0.478 |
| Shift 4 | XGBoost | 0.499 | 0.349 | 0.508 |

Bold values indicate the lowest ECE among the five models at that shift level.

Several patterns are visible in the raw results.

First, Logistic Regression has the lowest ECE at every shift level. Its ECE values are 0.024, 0.156, 0.088, 0.038, and 0.031 from Shift 0 through Shift 4. Its accuracy, however, changes substantially, decreasing from 0.705 at Shift 0 to 0.640 at Shift 1 and then increasing to 0.894, 0.960, and 0.969 at Shifts 2–4. The increase in accuracy occurs in evaluation populations whose churn rate falls as tenure and contract commitment increase.

Second, SVM accuracy deteriorates sharply under the stronger shifts. It decreases from 0.691 at Shift 0 to 0.259 at Shift 2, 0.040 at Shift 3, and 0.031 at Shift 4. At the same time, its ECE increases from 0.055 at Shift 0 to 0.395, 0.470, and 0.478 at Shifts 2–4. Thus, in this experiment, the SVM's confidence estimates become increasingly inconsistent with observed outcomes as the shifted populations become more severe.

Third, Gaussian Naive Bayes has the largest ECE at every reported shift level. Its ECE increases from 0.290 at Shift 0 to 0.638 at Shift 4, while its accuracy decreases from 0.674 to 0.362. This indicates poor calibration in the in-distribution control and further deterioration under the severe shifted population.

Finally, Random Forest and XGBoost exhibit intermediate but non-monotonic patterns. Their ECE values increase substantially at the stronger shift levels, although their accuracy patterns differ from one another.

### 5.2 Reliability diagrams

Figure 1 displays the reliability diagrams for the five classifiers at each shift level. The dashed diagonal represents perfect calibration.

At Shift 0, most model curves are comparatively close to the diagonal, although Naive Bayes shows larger deviations. At Shifts 3 and 4, the SVM and XGBoost curves flatten near the lower part of the plot, indicating low observed positive-class frequencies across much of the predicted-probability range. Logistic Regression remains visually closest to the diagonal across the evaluated shift levels.

The reliability diagrams support the numerical ECE results, but they should be interpreted as finite-sample, binned summaries rather than exact demonstrations of calibration or miscalibration at every probability value.

---

## 6. Statistical Analysis

### 6.1 Pairwise model comparisons

Paired bootstrap \(t\)-tests and Wilcoxon signed-rank tests were computed for every pair of models at every shift level, producing 25 pairwise model comparisons in the reported analysis.

Every pairwise comparison was statistically significant, with \(p<0.001\) in all cases and \(p<10^{-20}\) in the large majority. These results indicate that the paired bootstrap ECE values differed consistently between the compared models under the testing procedure used.

At Shift 4, for example, the comparison between Logistic Regression and SVM produced a paired \(t\)-statistic of \(-736.4\) with \(p\approx 0\), and a Wilcoxon statistic of 0.0 with \(p\approx 0\). The corresponding ECE values were 0.031 for Logistic Regression and 0.478 for SVM.

The reported tests concern the paired bootstrap distributions and should not be interpreted as proving that the observed model rankings would be identical under every dataset, sampling design, or shift mechanism.

### 6.2 Shift magnitude and calibration error

Pooling all model–shift observations, the association between PSI and ECE was positive and statistically significant:

- Pearson correlation: \(r=0.512\), \(p=0.0089\);
- Spearman correlation: \(\rho=0.573\), \(p=0.0028\).

These values indicate a moderate positive association between the constructed shift measure and calibration error across the pooled observations.

When correlations were calculated separately for each model, they were not statistically significant at conventional thresholds. For example, Logistic Regression had Pearson \(r=0.254\) and \(p=0.681\). Each model-specific correlation is based on only five shift levels, so these analyses have limited statistical power. The pooled analysis has more observations, but pooling also combines different model families and therefore describes an overall association rather than a model-specific law.

---

## 7. Discussion

### 7.1 Main findings

The central empirical result is that calibration behavior differs substantially across classifiers under the specified distribution-shift design. The predicted probabilities do not retain a uniform probabilistic interpretation simply because the classifiers perform well on the in-distribution control. Logistic Regression has the lowest ECE at all five shift levels, whereas Gaussian Naive Bayes, SVM, and XGBoost show substantially larger calibration errors under the stronger shifts.

This result answers the paper's title question in a model- and distribution-dependent way. Confidence scores remain comparatively reliable in the evaluated setting when the model's predicted probabilities continue to agree with observed frequencies on the shifted population. In the present experiment, this agreement is strongest for Logistic Regression as measured by ECE. It is not a universal property of classifier confidence scores: the SVM and Naive Bayes results demonstrate that scores can become poorly aligned with observed outcomes under the same shift design.

The findings are consistent with the broader calibration literature, which treats model scores as quantities requiring empirical validation rather than automatically valid probabilities. They also align with prior work showing that calibration can deteriorate under dataset shift and that in-distribution calibration is not sufficient evidence of out-of-distribution reliability (Ovadia et al., 2019).

### 7.2 Calibration and accuracy are distinct

The Logistic Regression results provide a clear example of the distinction between accuracy and calibration. Logistic Regression's accuracy falls at Shift 1 and then rises sharply at Shifts 2–4, while its ECE changes in a different pattern and remains the lowest among the models. Therefore, accuracy cannot be used as a substitute for calibration.

The increase in Logistic Regression accuracy at the stronger shifts should also be interpreted in the context of the shift design. The evaluation populations contain longer-tenure customers and customers with more committed contracts, and the churn rate falls across these populations. A classifier can achieve high accuracy in a population with a low event rate by predicting the majority outcome frequently. Such accuracy does not by itself establish that its predicted probabilities are calibrated. In this experiment, the comparatively low Logistic Regression ECE supplies additional evidence about probability–frequency agreement, whereas accuracy alone would not.

The Brier scores provide a complementary probabilistic measure. Logistic Regression has a Brier score of 0.031 at Shift 4, while its ECE is 0.031. The Brier score should nevertheless not be read as a pure calibration statistic because it also depends on the informativeness and concentration of the predictions.

### 7.3 Interpreting the model differences

The results suggest that the five classifiers respond differently to the same shifted populations. Logistic Regression was trained within a pipeline designed for probability estimation through a probabilistic classification objective, while the SVM is fundamentally a margin-based classifier whose decision function is transformed into probability estimates when probability estimation is enabled. The observed SVM pattern—near-zero accuracy and high ECE at Shifts 3 and 4—is consistent with a situation in which the learned decision rule performs poorly on the shifted population while its probability transformation does not adequately signal that failure.

This interpretation should be stated cautiously. The present experiment does not isolate the contribution of the RBF decision function from the contribution of the SVM probability-estimation procedure. The probability estimates were obtained with probability estimation enabled, and the paper does not compare them with alternative calibration methods or with the untransformed decision scores. Consequently, the results establish the observed behavior of this SVM implementation under this shift design, not a universal property of all SVM probability estimates.

The Naive Bayes results likewise show severe calibration error even at the in-distribution control and greater error at the severe shift. A possible interpretation is that the model's assumptions about the feature distribution and conditional independence do not produce reliable probabilities for this dataset. However, the present analysis does not directly test those assumptions. The safest conclusion is empirical: Gaussian Naive Bayes had the largest ECE at every reported shift level and its ECE increased from 0.290 to 0.638.

Random Forest and XGBoost occupy an intermediate position in the results, but their patterns are not monotonic in every metric. This prevents a simple claim that all tree-based models degrade in the same way. Their behavior supports the broader conclusion that calibration under shift should be assessed separately for each model and evaluation population.

### 7.4 Shift magnitude and pooled association

The pooled Pearson and Spearman results provide evidence of a positive association between the chosen shift measure and ECE across the 25 model–shift observations. This supports the interpretation that more divergent evaluation populations tend, in the pooled analysis, to be associated with larger calibration error.

The result should not be interpreted as a universal monotonic relationship for every model. Model-specific correlations were not statistically significant, and the Logistic Regression ECE values themselves are non-monotonic across the five shift levels. The pooled association therefore summarizes the combined behavior of several models rather than proving that ECE must increase with PSI within every classifier.

The distinction between pooled and model-specific analysis is statistically important. Pooling increases the number of observations used to estimate the overall association, but it also combines observations with different model identities. A pooled correlation can therefore be statistically significant even when no individual model-specific correlation reaches significance. In this study, the small number of shift levels per model limits the strength of conclusions about model-specific trends.

### 7.5 Theoretical interpretation of "remaining probabilities"

The phrase "remain probabilities" can be made precise operationally. A predicted score retains its probabilistic interpretation on a target population when the score agrees with the relevant conditional outcome frequency on that population. In finite data, this agreement is estimated using tools such as reliability diagrams, ECE, and proper probabilistic scores.

However, a low ECE on the evaluated sample does not establish perfect calibration in the underlying population. ECE depends on binning, sample size, and the particular target population. Likewise, a single shifted dataset cannot establish that a model will remain calibrated under arbitrary future changes. The strongest supported conclusion is therefore conditional: under the shift design used here, Logistic Regression's probabilities remained comparatively well aligned with observed outcomes, whereas the other models—especially SVM and Naive Bayes—showed considerably greater misalignment.

The experiment also illustrates why calibration should be treated as a property of the combination of model, score-generation procedure, and evaluation distribution. It is not solely a fixed attribute of the classifier name. The same model may be calibrated in one population and miscalibrated in another.

---

## 8. Limitations

Several limitations constrain the interpretation of the results.

1. **Dataset and domain specificity.** The experiment uses one IBM Telco Customer Churn dataset and one business domain. The results may not generalize to other populations, tasks, or feature structures.

2. **Combined shift design.** Contract type and tenure change together across the evaluation populations, and the churn rate changes as well. The study therefore does not isolate pure covariate shift from pure prior-probability shift.

3. **Limited number of shift levels.** Each model-specific shift–ECE correlation is based on only five observations. This limits statistical power and makes the individual correlation estimates unstable.

4. **Classical tabular models only.** The analysis evaluates Logistic Regression, Gaussian Naive Bayes, Random Forest, SVM, and XGBoost. The findings should not be generalized directly to neural networks or other uncertainty-estimation methods.

5. **SVM probability estimation.** The SVM uses probability estimates enabled in the implementation. Some observed miscalibration may depend on this particular probability-estimation procedure rather than on the SVM decision function in isolation.

6. **ECE estimation.** ECE is bin-dependent and is subject to finite-sample estimation error. Reliability diagrams and Brier scores provide useful complementary information, but none of these measures alone establishes population-level calibration.

7. **Statistical testing scope.** The reported pairwise tests concern the paired bootstrap procedure used in the analysis. The paper does not report a correction for the family of pairwise comparisons, nor does it establish that the observed differences persist under alternative resampling or multiple-testing procedures.

8. **No recalibration experiment.** The study evaluates the scores produced by the trained models but does not test whether temperature scaling, isotonic regression, prior-shift correction, or another recalibration procedure would restore calibration under the shifted populations.

---

## 9. Conclusion

This paper examined whether classifier confidence scores retain their probabilistic meaning under a structured distribution shift in the IBM Telco Customer Churn dataset. The results show that they do not do so uniformly across the five evaluated models. Logistic Regression had the lowest ECE at every shift level, while Gaussian Naive Bayes and the RBF-kernel SVM showed severe calibration deterioration under the stronger shifts; the SVM's accuracy fell to 0.031 and its ECE rose to 0.478 at Shift 4.

Across all model–shift observations, the chosen PSI measure was positively associated with ECE, although model-specific correlations were not statistically significant with only five shift levels per model. The evidence therefore supports an overall association between the constructed distributional divergence and calibration error, but not a universal model-by-model monotonic rule.

The answer to the paper's central question is consequently conditional: confidence scores remain comparatively interpretable as probabilities only when their agreement with observed frequencies is verified on the target population, and that agreement depends on both the classifier and the nature of the shift. In-distribution accuracy or calibration should not be treated as sufficient evidence that deployed probabilities will remain reliable.

---

## References

DeGroot, M. H., & Fienberg, S. E. (1983). The comparison and evaluation of forecasters. *The Statistician, 32*(1–2), 12–22. https://doi.org/10.2307/2987588

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In D. Precup & Y. W. Teh (Eds.), *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. In T. G. Dietterich, S. Becker, & Z. Ghahramani (Eds.), *Advances in Neural Information Processing Systems 14* (pp. 841–848). MIT Press.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). Association for Computing Machinery. https://doi.org/10.1145/1102351.1102430

Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J. V., Lakshminarayanan, B., & Snoek, J. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. In H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, & R. Garnett (Eds.), *Advances in Neural Information Processing Systems 32* (pp. 13991–14002). Curran Associates. https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). (2009). *Dataset shift in machine learning*. MIT Press. https://doi.org/10.7551/mitpress/9780262170055.001.0001

---

## Appendix: Reproducibility

- **Dataset:** IBM Telco Customer Churn (`telco.csv`)
- **Pipeline:** `src/calibration_shift_pipeline.py`
- **Raw results:** `results/results_summary.csv`
- **Pairwise comparisons:** `results/pairwise_model_comparisons.csv`
- **Shift–ECE correlations:** `results/shift_vs_ece_correlation.csv`
- **Figure:** `figures/reliability_diagrams.png`