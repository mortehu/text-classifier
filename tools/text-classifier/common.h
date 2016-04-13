#ifndef TOOLS_TEXT_CLASSIFIER_COMMON_H_
#define TOOLS_TEXT_CLASSIFIER_COMMON_H_ 1

struct Header {
  float class_id;
  uint32_t hash_count;
};

enum CostFunction {
  kCostFunctionDefault,
  kCostFunctionAUROC,
  kCostFunctionRMSE,
  kCostFunctionFn,
};

enum Mode {
  kModeRegression,
  kModeClassification,
};

enum WeightType { kWeightBNS, kWeightIDF, kWeightLogOdds, kWeightNone };

struct TextClassifierParams {
  Mode mode = kModeClassification;

  // Shuffle documents before dividing into folds for cross-validation.
  bool do_shuffle = true;

  CostFunction cost_function = kCostFunctionDefault;
  double cost_parameter = 1.0;

  // Distance at which regression errors are discounted.
  float regression_epsilon = 0.1f;

  // Completion criteria for optimization.
  float epsilon = 1.0f;

  // Maximum value of class 0.
  float class_threshold = 0.0f;

  // Weight of intercept column.
  float intercept = 0.0f;

  // User selected C value.  If non-zero, grid-search is skipped.
  float param_c = 0.0f;

  // User selected minimum C value.
  float param_min_c = 0.0f;

  // User selected class weight ratio.
  float weight_ratio = 0.0f;

  // Value to substitute for zero when counting feature occurrences during
  // weighting.
  double min_count = 0.1;

  // Max number of iterations in training loop.
  size_t max_iterations = 10000000;

  // Set to true to print debug information.
  bool do_debug = true;

  uint64_t threshold = 2;

  bool do_normalize = true;
  WeightType weight_type = kWeightLogOdds;
};

#endif  // !TOOLS_TEXT_CLASSIFIER_COMMON_H_
