1. Run Proxy Score Ablation Study / Weighting of normalized SynFlow/NASWOT sum

2. Improve based on results of previous runs and Proxy Score result

3. Run some kind of HPO or try different HP's, interesting candidates to tune:
For an unseen-data challenge, I would prioritize hyperparameters that adapt to cheap dataset diagnostics and avoid aggressive semantic assumptions. Augmentation should be
  conservative and conditional—not treated like an ordinary image-classification benchmark.

  My recommended top 10 are:

  1. Base learning rate

     Probably the most important training HP. It is currently coupled to batch size.

     Suggested space: log-uniform 0.005–0.15, or tune the coefficient in the current scaling rule.

     Location: submission_template/trainer.py:30

  2. Weight decay

     Crucial for balancing underfitting and overfitting across unknown dataset sizes and model capacities.

     Suggested space: log-uniform 1e-5–5e-3.

     Location: submission_template/trainer.py:37

  3. Initial channel width

     This is the strongest direct control over model capacity, memory consumption, and speed.

     Suggested choices: {16, 24, 32, 48, 64}, constrained by dataset size and resolution.

     Location: submission_template/nas.py:104

  4. Number of cells

     Controls depth and interacts strongly with width. Tune width and depth jointly because a deep-wide network can consume the budget before converging.

     Suggested choices: {2, 3, 4, 5, 6}.

     Location: submission_template/nas.py:104

  5. Dropout rate

     Useful for unseen tasks because the appropriate model capacity is uncertain. It should depend on the sample count and model size.

     Suggested space: 0.0–0.4, with lower values for large datasets and underfitting-prone configurations.

     Location: submission_template/nas.py:109

  6. Label smoothing

     Usually safer than geometry-based augmentation because it does not modify the input. Nevertheless, too much smoothing can hurt fine-grained or noisy-label tasks.

     Suggested choices: {0.0, 0.025, 0.05, 0.1}.

     Location: submission_template/trainer.py:83

  7. Augmentation gate and probability

     Treat this as one conditional HP:

     augmentation mode ∈ {none, noise, occlusion, noise+occlusion}
     probability ∈ [0.0, 0.3]

     I would keep translation, rotation, flips, and color transformations disabled unless diagnostics provide strong evidence that they preserve the label. Vertical flips
     are especially dangerous for digits, characters, medical data, directional objects, and encoded arrays.

     The current encoded_likely → no augmentation gate is sensible, but the threshold defining encoded data may also need calibration.

     Location: submission_template/data_processor.py:261

  8. Batch size

     Batch size affects optimization, learning-rate scaling, memory, and the number of updates possible within the time budget.

     Suggested choices: {16, 32, 64}, with 128 only when images are small and memory permits.

     Heuristic: submission_template/helpers.py:760

  9. Short-training fidelity

     This includes the number of updates and validation examples used to select finalists. If the fidelity is too low, the search may consistently select architectures that
     learn quickly but finish poorly.

     Most important components:
      - Maximum updates: currently 40
      - Short-training learning rate: currently 0.03
      - Validation subset: currently 1,024 examples
      - Total finalist-training budget

     Location: submission_template/nas.py:325

  10. Class-imbalance weighting policy

  This is particularly relevant for unseen datasets. The current code activates inverse-frequency weights at an imbalance ratio of 3, but full inverse-frequency weighting
  can overcorrect severely.

  Suggested conditional choices:

     weighting ∈ {none, inverse-sqrt frequency, inverse frequency}
     activation threshold ∈ {2, 3, 5, 10}

  I would expect inverse-square-root weighting to be safer across unknown datasets.

  Location: submission_template/trainer.py:91

  A compact first BO space could therefore be:

  learning_rate          loguniform(0.005, 0.15)
  weight_decay           loguniform(1e-5, 5e-3)
  init_channels          {16, 24, 32, 48, 64}
  n_cells                {2, 3, 4, 5, 6}
  dropout                uniform(0.0, 0.4)
  label_smoothing        {0.0, 0.025, 0.05, 0.1}
  augmentation_mode      {none, noise, occlusion, noise+occlusion}
  augmentation_prob      uniform(0.0, 0.3), conditional
  batch_size             {16, 32, 64}
  class_weighting        {none, inverse_sqrt, inverse}

  I would initially hold the cell topology, proxy weighting, candidate count, and diversity coefficient fixed. Otherwise BO must optimize architecture, training, and search
  reliability simultaneously, which makes the observed validation score substantially noisier. After finding a robust training policy, architecture-search parameters can
  form a second optimization stage.

