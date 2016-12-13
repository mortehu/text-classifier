#ifndef TOOLS_TEXT_CLASSIFIER_SVM_H_
#define TOOLS_TEXT_CLASSIFIER_SVM_H_ 1

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <cctype>
#include <cinttypes>
#include <cmath>
#include <map>

#include <kj/debug.h>
#include <kj/io.h>
#include <sparsehash/dense_hash_map>

#include "base/stringref.h"
#include "tools/text-classifier/common.h"
#include "tools/text-classifier/model.h"

typedef uint8_t HashCountType;

namespace ev {

class TextClassifierSVMModel : public TextClassifierModel {
 public:
  struct Document {
    size_t first_idx;
    uint32_t hash_count;
    float class_id;
  };

  struct Stat {
    uint32_t true_positives = 0;
    uint32_t false_positives = 0;
  };

  TextClassifierSVMModel(TextClassifierParams params);

  void Train(cantera::ColumnFileReader input) final;

  void Save(kj::AutoCloseFd output) const final;

  void Load(kj::AutoCloseFd input) final;

  void Print() const final;

  double Classify(const std::vector<uint64_t>& features) const final;

 private:
  typedef std::vector<std::tuple<float, HashCountType>> HashWeights;

  // Returns scaling weights and count thresholds for each feature.
  //
  // See "BNS Feature Scaling: An Improved Representation over TF·IDF for SVM
  // Text Classification" by George Forman.
  HashWeights GetHashWeights(const std::vector<size_t>& training_set);

  void Solve();

  float Dot(size_t idx, const std::vector<float>& w,
            const HashWeights& feature_weights);

  void AddScaled(std::vector<float>& w, size_t idx, float scale,
                 const HashWeights& feature_weights);

  std::pair<float, bool> TestParameters(float C_pos, float C_neg,
                                        size_t shard_count, size_t max_iter);

  float SolveSVC(float C_pos, float C_neg, size_t max_iter, size_t shard_idx,
                 size_t shard_count, std::vector<float>* result_w = nullptr,
                 HashWeights* result_weights = nullptr);

  float SolveSVR(float C, size_t max_iter, size_t shard_idx, size_t shard_count,
                 std::vector<float>* result_w = nullptr,
                 HashWeights* result_weights = nullptr);

  google::dense_hash_map<uint64_t, uint32_t> hash_to_idx_;

  std::map<std::pair<double, double>, double> optimize_cache_;

  std::pair<double, double> optimize_minimum_{0.0, 0.0};
  float optimize_best_score_ = HUGE_VAL;

  // Number of unique indexes.
  size_t index_count_ = 0;

  std::vector<uint32_t> doc_hash_idx_;

  // Feature occurrences in excess of 1.  If all features are unique within
  // each document, this vector is empty.
  std::vector<HashCountType> doc_hash_counts_;

  std::vector<Document> documents_;

  // Document order, used for stratified N-fold cross-validation.
  std::vector<size_t> document_order_;

  // Index of first element in `document_order_` whose class is positive.
  size_t class_split_ = SIZE_MAX;

  // Training target for classification.
  std::vector<int8_t> y_binary_;

  // Training target values for regression.
  std::vector<float> y_scalar_;

  // Stored alpha values, used to warm-start cross-validation.  Saved about 20%
  // computation time in a test.
  std::array<std::vector<float>, 10> shard_alpha_;
  std::array<HashWeights, 10> shard_feature_weights_;

  // Final solution.
  std::vector<float> w_;
  HashWeights feature_weights_;

  size_t n_positive_ = 0;
  size_t n_negative_ = 0;

  float C_min_ = 0.0f;
  float C_max_ = 65536.0f;

  TextClassifierParams params_;

  // Final model.
  float intercept_weight_ = 0.0f;
  float bias_ = 0.0f;
  google::dense_hash_map<uint64_t, float> decision_plane_;
  google::dense_hash_map<uint64_t, std::pair<float, HashCountType>>
      classify_feature_weights_;
};

}  // namespace ev

#endif  // !TOOLS_TEXT_CLASSIFIER_SVM_H_
