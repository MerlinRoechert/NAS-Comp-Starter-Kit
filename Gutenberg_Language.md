# Gutenberg and Language: structural correction

Gutenberg and Language are not small-data tasks: they contain roughly
45,000–50,000 training examples. Their distinguishing feature is that inputs
are sparse binary character-by-position grids. Rows encode character identity
and columns encode sequence position.

The previous natural-image architecture repeatedly downsampled these small
grids and finally applied global average pooling. This encouraged translation
invariance and removed absolute positional information. At the same time,
training-free proxies could prefer large models even though their structural
bias was unsuitable.

The pipeline now detects this pattern without using dataset names. It requires
a single-channel, low-cardinality, sparse grid with approximately one active
entry per row or column, so either sequence-axis orientation is supported. For
detected sequence grids it:

- retains the existing no-augmentation policy for encoded inputs;
- permits at most one spatial downsampling;
- flattens the remaining spatial map instead of globally averaging positions;
- searches smaller widths and depths with a three-million-parameter cap;
- mildly penalizes proxy scores above one million parameters;
- uses the same position-sensitive models during proxy calibration.

Conventional image datasets retain the original architecture and search policy.
The diagnostic is printed as `sequence_grid=True`; confirm that it activates
for Gutenberg and Language before interpreting new results.
