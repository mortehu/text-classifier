#include <iostream>
#include "tools/text-classifier/svm.h"

#include "base/cat.h"
#include "base/error.h"
#include "base/file.h"
#include "base/macros.h"
#include "base/parallel-sort.h"
#include "base/random.h"
#include "base/string.h"
#include "base/stringref.h"
#include "base/thread-pool.h"
#include "tools/text-classifier/common.h"
#include "tools/text-classifier/table-storage.h"

namespace {

template <typename T>
T Pow2(T v) {
  return v * v;
}

// Approximates the inverse error function.
double InvErf(double x) {
  static const auto a = 8 * (M_PI - 3) / (3 * M_PI * (4 - M_PI));

  static const auto k2_pi_a = 2.0 / (M_PI * a);

  const auto kLn1MinusX2_2 = std::log(1.0 - x * x) / 2.0;

  auto result = std::sqrt(
      std::sqrt((k2_pi_a + kLn1MinusX2_2) * (k2_pi_a + kLn1MinusX2_2) -
                std::log(1.0 - x * x) / a) -
      (k2_pi_a + kLn1MinusX2_2));

  if (x < 0) result = -result;

  return result;
}

// Approximate inverse of the normal distribution's cumulative density function.
double InvNormalCDF(double p) { return M_SQRT2 * InvErf(2 * p - 1.0); }

}  // namespace

namespace ev {

TextClassifierSVMModel::TextClassifierSVMModel(TextClassifierParams params)
    : params_(params) {
  decision_plane_.set_empty_key(0);
  classify_feature_weights_.set_empty_key(0);
}

void TextClassifierSVMModel::Train(ev::TableReader reader) {
  std::vector<uint64_t> all_hashes;

  // If the input is larger than main memory, it won't all fit in
  // `all_hashes`.  Therefore, we keep track of the most commonly occurring
  // hashes in this array, so that we can avoid inserting them into
  // `all_hashes` once they have reached their thresholds.  Empirically, this
  // reduces the size of `all_hashes` by about 91%.
  const auto kCommonHashesSize = 0x1000000;
  std::vector<std::pair<uint64_t, uint8_t>> common_hashes;
  common_hashes.resize(kCommonHashesSize);

  size_t document_count = 0;

  std::tuple<std::string, std::string> row;

  while (reader.GetRow(row)) {
    const auto& header_data = std::get<0>(row);
    KJ_REQUIRE(header_data.size() == sizeof(Header), header_data.size());
    const auto& payload_data = std::get<1>(row);
    KJ_REQUIRE((payload_data.size() % sizeof(uint64_t)) == 0,
               payload_data.size());

    const auto& header = *reinterpret_cast<const Header*>(header_data.data());

    if (params_.mode == kModeRegression)
      y_scalar_.emplace_back(header.class_id);

    if (header.class_id <= params_.class_threshold) {
      if (params_.mode == kModeClassification) y_binary_.emplace_back(-1);
      ++n_negative_;
    } else {
      if (params_.mode == kModeClassification) y_binary_.emplace_back(1);
      ++n_positive_;
    }

    auto hashes = reinterpret_cast<const uint64_t*>(payload_data.data());

    uint64_t last_hash = 0;

    for (size_t i = 0; i < header.hash_count; ++i) {
      const auto hash = hashes[i];

      if (hash == last_hash) continue;
      last_hash = hash;

      const auto cache_key = hash % kCommonHashesSize;

      //  Only register the hash if it hasn't already reached the threshold in
      //  `common_hashes`.
      if (common_hashes[cache_key].first != hash) {
        all_hashes.emplace_back(hash);
        if (common_hashes[cache_key].second < params_.threshold) {
          common_hashes[cache_key].first = hash;
          common_hashes[cache_key].second = 1;
        }
      } else if (common_hashes[cache_key].second < params_.threshold) {
        all_hashes.emplace_back(hash);
        ++common_hashes[cache_key].second;
      }
    }

    ++document_count;
  }

  common_hashes.clear();
  common_hashes.shrink_to_fit();

  ev::ThreadPool thread_pool;
  ev::ParallelSort(all_hashes.begin(), all_hashes.end(), std::less<uint64_t>(),
                   thread_pool);

  size_t passing_features = 0, total_features = 0;

  for (size_t i = 0; i < all_hashes.size(); ++i) {
    if ((i + 1) == all_hashes.size() || all_hashes[i + 1] != all_hashes[i]) {
      ++total_features;
      if (i + 1 >= params_.threshold &&
          all_hashes[i] == all_hashes[i - params_.threshold + 1])
        all_hashes[passing_features++] = all_hashes[i];
    }
  }

  if (params_.do_debug) {
    fprintf(stderr, "%zu of %zu features pass threshold.\n", passing_features,
            total_features);
  }

  all_hashes.resize(passing_features);
  all_hashes.shrink_to_fit();

  hash_to_idx_.set_empty_key(0);
  hash_to_idx_.resize(all_hashes.size());

  for (size_t i = 0; i < all_hashes.size(); ++i)
    hash_to_idx_[all_hashes[i]] = i + 1;

  all_hashes.clear();
  all_hashes.shrink_to_fit();

  if (params_.mode == kModeRegression) {
    // Centering the response value around the mean improves accuracy in SVR.
    // It's similar to applying class weights in SVC.
    //
    // TODO(mortehu):
    //   Check out https://cran.r-project.org/web/packages/LambertW/
    //
    // TODO(mortehu): This fucks with cross-validation.

    KJ_ASSERT(y_binary_.empty());

    double sum = 0.0;
    for (auto y : y_scalar_) sum += y;
    bias_ = sum / y_scalar_.size();
    for (auto& y : y_scalar_) y -= bias_;
  }

  index_count_ = hash_to_idx_.size() + 1;

  reader.SeekToStart();

  while (reader.GetRow(row)) {
    const auto& header_data = std::get<0>(row);
    const auto& payload_data = std::get<1>(row);
    const auto& header = *reinterpret_cast<const Header*>(header_data.data());

    Document doc;
    doc.first_idx = doc_hash_idx_.size();
    doc.hash_count = header.hash_count;
    doc.class_id = header.class_id;

    uint64_t last_hash = 0;

    auto hashes = reinterpret_cast<const uint64_t*>(payload_data.data());

    for (size_t i = 0; i < header.hash_count; ++i) {
      const auto hash = hashes[i];

      auto j = hash_to_idx_.find(hash);
      if (j == hash_to_idx_.end()) {
        --doc.hash_count;
        continue;
      }

      if (hash == last_hash) {
        while (doc_hash_counts_.size() < doc_hash_idx_.size())
          doc_hash_counts_.emplace_back(0);
        if (doc_hash_counts_.back() < std::numeric_limits<HashCountType>::max())
          ++doc_hash_counts_.back();
        --doc.hash_count;
      } else {
        doc_hash_idx_.emplace_back(j->second);
        last_hash = hash;
      }
    }

    if (!doc.hash_count) continue;

    documents_.emplace_back(doc);
  }

  // Make sure `doc_hash_counts_` vector is equal in size to `doc_hash_idx_`.
  if (!doc_hash_counts_.empty()) doc_hash_counts_.resize(doc_hash_idx_.size());

  if (params_.param_c) {
    C_min_ = params_.param_c;
    C_max_ = params_.param_c;
  } else {
    uint32_t max_count = 0;
    for (const auto& document : documents_)
      max_count = std::max(max_count, document.hash_count);
    C_min_ = 1.0f / (2.0f * documents_.size() * max_count);
    // Round down to nearest power of two.
    C_min_ = std::floor(std::log(C_min_) / std::log(2.0f));
    // Clamp to a reasonable lower bound.
    C_min_ = std::max(params_.do_normalize ? -10.0f : -16.0f, C_min_);

    C_min_ = std::max(params_.param_min_c, std::pow(2.0f, C_min_));
  }

  Solve();

  for (const auto& p : hash_to_idx_) {
    const auto w = w_[p.second];

    if (w) {
      KJ_ASSERT(std::isfinite(w), w, p.first);

      if (p.first == 0)
        intercept_weight_ = w;
      else
        decision_plane_[p.first] = w;
    }

    const auto weight = std::get<float>(feature_weights_[p.second]);

    if (weight) {
      KJ_ASSERT(std::isfinite(weight), weight, p.first);

      const auto hash_count_threshold =
          std::get<HashCountType>(feature_weights_[p.second]);

      classify_feature_weights_[p.first] =
          std::make_pair(weight, hash_count_threshold);
    }
  }

  w_.clear();
  w_.shrink_to_fit();

  feature_weights_.clear();
  feature_weights_.shrink_to_fit();
}

void TextClassifierSVMModel::Solve() {
  size_t shard_count = 10;
  if (params_.mode == kModeClassification) {
    shard_count = std::max(
        2_z, std::min(n_positive_ / 2, std::min(n_negative_ / 2, 10_z)));
  }

  document_order_.clear();

  for (size_t i = 0; i < documents_.size(); ++i)
    document_order_.emplace_back(i);

  if (params_.do_shuffle) {
    std::shuffle(document_order_.begin(), document_order_.end(),
                 std::mt19937_64(1234));
  }

  auto partition_point = std::stable_partition(
      document_order_.begin(), document_order_.end(), [this](const auto idx) {
        return documents_[idx].class_id <= params_.class_threshold;
      });
  class_split_ = partition_point - document_order_.begin();

  if (C_min_ != C_max_) {
    optimize_best_score_ =
        (params_.cost_function == kCostFunctionRMSE) ? HUGE_VAL : 0.0f;

    float C_pos, C_neg;

    if (params_.mode == kModeRegression) {
      C_pos = C_neg = C_min_;
    } else if (!params_.weight_ratio) {
      C_pos = 2.0f * C_min_ * n_negative_ / (n_positive_ + n_negative_);
      C_neg = 2.0f * C_min_ * n_positive_ / (n_positive_ + n_negative_);
    } else {
      C_pos = C_min_;
      C_neg = C_min_ / params_.weight_ratio;
    }

    // The sequence in which we try scaling `C_pos` and `C_neg`.
    static const std::array<std::pair<float, float>, 5> kGrowthFactors{
        {{2.0f, 2.0f}, {1.0f, 0.5f}, {0.5f, 1.0f}, {1.0f, 2.0f}, {2.0f, 1.0f}}};

    size_t growth_factor_idx = 0;

    // The score representing a perfect outcome.  If this is achieved, we can
    // stop tuning hyperparameters.
    const auto optimum =
        (params_.cost_function == kCostFunctionRMSE) ? 0.0 : 1.0;

    // Set to true if we've seen an improvement in this round, so we should
    // retry the scaling factors from the start before giving up.
    bool do_reset = false;

    for (;;) {
      const auto result =
          TestParameters(C_pos, C_neg, shard_count, params_.max_iterations);
      if (result.second) {
        if (result.first == optimum) break;
        if (growth_factor_idx > 0) do_reset = true;
      } else {
        if (params_.mode == kModeRegression) break;

        if (++growth_factor_idx == kGrowthFactors.size()) {
          if (!do_reset) break;
          growth_factor_idx = 0;
          do_reset = false;
        }

        C_pos = optimize_minimum_.first;
        C_neg = optimize_minimum_.second;
      }

      C_pos *= kGrowthFactors[growth_factor_idx].first;
      C_neg *= kGrowthFactors[growth_factor_idx].second;
    }

    for (auto& w : shard_feature_weights_) {
      w.clear();
      w.shrink_to_fit();
    }

    for (auto& a : shard_alpha_) {
      a.clear();
      a.shrink_to_fit();
    }
  } else if (params_.weight_ratio) {
    optimize_minimum_.first = C_min_;
    optimize_minimum_.second = C_min_ / params_.weight_ratio;
  } else {
    optimize_minimum_.first =
        2.0f * C_min_ * n_negative_ / (n_positive_ + n_negative_);
    optimize_minimum_.second =
        2.0f * C_min_ * n_positive_ / (n_positive_ + n_negative_);
  }

  if (params_.mode == kModeRegression) {
    SolveSVR(optimize_minimum_.first, params_.max_iterations, 0, 1, &w_,
             &feature_weights_);
  } else {
    KJ_ASSERT(params_.mode == kModeClassification);
    SolveSVC(optimize_minimum_.first, optimize_minimum_.second,
             params_.max_iterations, 0, 1, &w_, &feature_weights_);
  }
}

void TextClassifierSVMModel::Save(kj::AutoCloseFd output) const {
  static const auto kBufferLimit = 1024 * 1024;
  std::string buffer;

  auto buffer_append = [&buffer](auto v) {
    buffer.append(reinterpret_cast<const char*>(&v), sizeof(v));
  };

  uint64_t feature_weight_count = 0;
  for (const auto& p : classify_feature_weights_) {
    if (std::get<float>(p.second)) ++feature_weight_count;
  }

  buffer_append(static_cast<uint64_t>(decision_plane_.size()));
  buffer_append(feature_weight_count);
  buffer_append(bias_);

  if (params_.intercept != 0) {
    buffer_append(uint64_t(0));
    buffer_append(intercept_weight_);
  }

  for (const auto& p : decision_plane_) {
    const auto weight = std::get<float>(p);

    KJ_ASSERT(std::isfinite(weight), weight);

    buffer_append(p.first);
    buffer_append(weight);

    if (buffer.size() >= kBufferLimit) {
      ev::WriteAll(output, buffer);
      buffer.clear();
    }
  }

  for (const auto& p : classify_feature_weights_) {
    const auto weight = std::get<float>(p.second);

    if (!weight) continue;

    const auto hash_count_threshold = std::get<HashCountType>(p.second);

    KJ_ASSERT(std::isfinite(weight), weight);

    buffer_append(p.first);
    buffer_append(weight);
    buffer_append(hash_count_threshold);

    if (buffer.size() >= kBufferLimit) {
      ev::WriteAll(output, buffer);
      buffer.clear();
    }
  }

  ev::WriteAll(output, buffer);
}

void TextClassifierSVMModel::Load(kj::AutoCloseFd input) {
  int dupfd;
  KJ_SYSCALL(dupfd = dup(input));
  UniqueFILE model_input(fdopen(dupfd, "r"), fclose);

  uint64_t decision_plane_size = 0;
  uint64_t feature_weight_count = 0;

  KJ_REQUIRE(1 == fread(&decision_plane_size, sizeof(decision_plane_size), 1,
                        model_input.get()));
  KJ_REQUIRE(1 == fread(&feature_weight_count, sizeof(feature_weight_count), 1,
                        model_input.get()));
  KJ_REQUIRE(1 == fread(&bias_, sizeof(bias_), 1, model_input.get()));

  decision_plane_.resize(decision_plane_size);

  for (uint64_t i = 0; i < decision_plane_size; ++i) {
    uint64_t hash = 0;
    float weight = 0.0f;

    KJ_REQUIRE(1 == fread(&hash, sizeof(hash), 1, model_input.get()));
    KJ_REQUIRE(1 == fread(&weight, sizeof(weight), 1, model_input.get()));

    if (!hash) {
      intercept_weight_ = weight;
    } else {
      decision_plane_[hash] = weight;
    }
  }

  classify_feature_weights_.resize(feature_weight_count);

  for (uint64_t i = 0; i < feature_weight_count; ++i) {
    uint64_t hash = 0;
    float weight = 0.0f;
    HashCountType hash_count_threshold = 0;

    KJ_REQUIRE(1 == fread(&hash, sizeof(hash), 1, model_input.get()));
    KJ_REQUIRE(1 == fread(&weight, sizeof(weight), 1, model_input.get()));
    KJ_REQUIRE(1 == fread(&hash_count_threshold, sizeof(hash_count_threshold),
                          1, model_input.get()));
    KJ_REQUIRE(hash != 0);

    classify_feature_weights_[hash] =
        std::make_pair(weight, hash_count_threshold);
  }
}

void TextClassifierSVMModel::Print() const {
  printf("bias: %s\n", ev::FloatToString(bias_).c_str());

  for (const auto& p : decision_plane_) {
    const auto hash = p.first;
    const auto weight = p.second;

    if (!hash) {
      printf("intercept: %s\n", ev::FloatToString(weight).c_str());
    } else {
      printf("%" PRIx64 ": %s\n", hash, ev::FloatToString(weight).c_str());
    }
  }

  for (const auto& p : classify_feature_weights_) {
    const auto hash = p.first;
    const auto weight = std::get<float>(p.second);
    const auto hash_count_threshold = std::get<HashCountType>(p.second);

    KJ_REQUIRE(hash != 0);

    printf("%" PRIx64 "_weight: %s\n", hash, ev::FloatToString(weight).c_str());
    if (hash_count_threshold != 0)
      printf("%" PRIx64 "_threshold: %u\n", hash, hash_count_threshold);
  }
}

double TextClassifierSVMModel::Classify(
    const std::vector<uint64_t>& features) const {
  float v = intercept_weight_;
  auto sum_sq = Pow2(params_.intercept);

  for (auto i = features.begin(); i != features.end();) {
    const auto hash = *i++;
    KJ_ASSERT(hash > 0);

    HashCountType count = 0;

    while (i != features.end() && *i == hash) {
      if (count < std::numeric_limits<HashCountType>::max()) ++count;
      ++i;
    }

    auto w = classify_feature_weights_.find(hash);
    if (w == classify_feature_weights_.end()) continue;

    if (count < std::get<HashCountType>(w->second)) continue;

    sum_sq += Pow2(std::get<float>(w->second));

    auto j = decision_plane_.find(hash);
    if (j != decision_plane_.end()) v += j->second * std::get<float>(w->second);
  }

  if (params_.do_normalize && sum_sq) v /= std::sqrt(sum_sq);

  return v + bias_;
}

TextClassifierSVMModel::HashWeights TextClassifierSVMModel::GetHashWeights(
    const std::vector<size_t>& training_set) {
  size_t n_positive = 0, n_negative = 0;
  for (const auto document_idx : training_set)
    ++((documents_[document_idx].class_id > params_.class_threshold)
           ? n_positive
           : n_negative);

  HashWeights result;
  result.reserve(index_count_);

  // Maximum and minimum values for `tpr` and `fpr`, used in BNS calculation.
  const auto pr_min = std::min(0.0005, 1.0 / (n_positive + n_negative));
  const auto pr_max = 1.0 - pr_min;

  const auto get_weight = [this, n_positive, n_negative, pr_min, pr_max](
      size_t true_positives, size_t false_positives) {
    if (true_positives + false_positives < params_.threshold) return 0.0f;

    const auto tp = static_cast<double>(true_positives);
    const auto fp = static_cast<double>(false_positives);
    const auto fn = n_positive - tp;
    const auto tn = n_negative - fp;
    const auto tpr = std::max(pr_min, std::min(pr_max, tp / n_positive));
    const auto fpr = std::max(pr_min, std::min(pr_max, fp / n_negative));

    float weight;

    switch (params_.weight_type) {
      case kWeightBNS:
        weight = InvNormalCDF(tpr) - InvNormalCDF(fpr);
        break;

      case kWeightIDF:
        weight = std::log((n_positive + n_negative) / (tp + fp));
        break;

      case kWeightLogOdds: {
        const auto num =
            std::max(tp, params_.min_count) * std::max(tn, params_.min_count);
        const auto den =
            std::max(fp, params_.min_count) * std::max(fn, params_.min_count);

        weight = std::log(num / den);
      } break;

      case kWeightNone:
        weight = 1.0;
        break;

      default:
        KJ_FAIL_ASSERT("Unknown weight_type", params_.weight_type);
    }

    KJ_ASSERT(std::isfinite(weight), weight, tp, fp, fn, tn);

    return weight;
  };

  if (!doc_hash_counts_.empty()) {
    // If we get here, we're not just looking at unique input features, so we
    // need to find a threshold value for each feature to convert it to a
    // binary feature.  We do this by trying every threshold value for every
    // feature, and picking the one that has the highest weight.

    std::vector<std::tuple<uint32_t, HashCountType, bool>> hash_counts;

    for (const auto document_idx : training_set) {
      const auto& document = documents_[document_idx];

      auto label = document.class_id > params_.class_threshold;

      for (size_t i = 0; i < document.hash_count; ++i) {
        hash_counts.emplace_back(doc_hash_idx_[document.first_idx + i],
                                 doc_hash_counts_[document.first_idx + i],
                                 label);
      }
    }

    std::sort(hash_counts.begin(), hash_counts.end(), [](const auto& lhs,
                                                         const auto& rhs) {
      if (std::get<uint32_t>(lhs) != std::get<uint32_t>(rhs))
        return std::get<uint32_t>(lhs) < std::get<uint32_t>(rhs);
      return std::get<HashCountType>(lhs) > std::get<HashCountType>(rhs);
    });

    for (size_t begin = 0; begin < hash_counts.size();) {
      const auto hash = std::get<uint32_t>(hash_counts[begin]);
      while (hash > result.size()) result.emplace_back(0.0f, 0);

      auto end = begin + 1;

      while (end != hash_counts.size() &&
             std::get<uint32_t>(hash_counts[end]) == hash)
        ++end;

      size_t true_positives = 0, false_positives = 0;

      auto best_weight = 0.0f;
      HashCountType best_hash_count_threshold = 0;

      while (begin != end) {
        const auto hash_count_threshold =
            std::get<HashCountType>(hash_counts[begin]);
        do {
          ++(std::get<bool>(hash_counts[begin]) ? true_positives
                                                : false_positives);
          ++begin;
        } while (begin != end &&
                 std::get<HashCountType>(hash_counts[begin]) ==
                     hash_count_threshold);

        const auto weight = get_weight(true_positives, false_positives);
        if (std::fabs(weight) > std::fabs(best_weight)) {
          best_weight = weight;
          best_hash_count_threshold = hash_count_threshold;
        }
      }

      result.emplace_back(best_weight, best_hash_count_threshold);
    }

    KJ_REQUIRE(result.size() == index_count_);
  } else {
    std::vector<Stat> hash_stats;
    hash_stats.resize(index_count_);

    for (const auto document_idx : training_set) {
      const auto& document = documents_[document_idx];

      for (size_t i = 0; i < document.hash_count; ++i) {
        const auto hash_idx = doc_hash_idx_[document.first_idx + i];

        if (document.class_id <= params_.class_threshold) {
          ++hash_stats[hash_idx].false_positives;
        } else {
          ++hash_stats[hash_idx].true_positives;
        }
      }
    }

    for (const auto& stat : hash_stats) {
      result.emplace_back(get_weight(stat.true_positives, stat.false_positives),
                          0);
    }
  }

  return result;
}

float TextClassifierSVMModel::Dot(size_t idx, const std::vector<float>& w,
                                  const HashWeights& feature_weights) {
  float result = w[0] * params_.intercept;

  const auto& doc = documents_[idx];

  if (doc_hash_counts_.empty()) {
    for (size_t i = 0; i < doc.hash_count; ++i) {
      const auto hash_idx = doc_hash_idx_[doc.first_idx + i];
      result += w[hash_idx] * std::get<float>(feature_weights[hash_idx]);
    }
  } else {
    for (size_t i = 0; i < doc.hash_count; ++i) {
      const auto hash_idx = doc_hash_idx_[doc.first_idx + i];
      const auto& fw = feature_weights[hash_idx];
      if (doc_hash_counts_[doc.first_idx + i] < std::get<HashCountType>(fw))
        continue;
      result += w[hash_idx] * std::get<float>(fw);
    }
  }

  return result;
}

void TextClassifierSVMModel::AddScaled(std::vector<float>& w, size_t idx,
                                       float scale,
                                       const HashWeights& feature_weights) {
  const auto& doc = documents_[idx];

  if (!scale) return;

  if (doc_hash_counts_.empty()) {
    for (size_t i = 0; i < doc.hash_count; ++i) {
      const auto hash_idx = doc_hash_idx_[doc.first_idx + i];
      w[hash_idx] += scale * std::get<float>(feature_weights[hash_idx]);
    }
  } else {
    for (size_t i = 0; i < doc.hash_count; ++i) {
      const auto hash_idx = doc_hash_idx_[doc.first_idx + i];
      const auto& fw = feature_weights[hash_idx];
      if (doc_hash_counts_[doc.first_idx + i] < std::get<HashCountType>(fw))
        continue;
      w[hash_idx] += scale * std::get<float>(fw);
    }
  }

  w[0] += scale * params_.intercept;
}

std::pair<float, bool> TextClassifierSVMModel::TestParameters(
    float C_pos, float C_neg, size_t shard_count, size_t max_iter) {
  if (C_pos <= 0 || C_neg <= 0) return std::make_pair(HUGE_VAL, false);

  const auto cache_key = std::make_pair(C_pos, C_neg);

  auto cache_i = optimize_cache_.find(cache_key);
  if (cache_i != optimize_cache_.end())
    return std::make_pair(cache_i->second, false);

  if (params_.do_debug) fprintf(stderr, "--C=%.6g --weight-ratio=%.6g ", C_pos, C_pos / C_neg);

  ev::ThreadPool thread_pool;

  std::vector<std::future<float>> score_promises;

  for (size_t shard_idx = 0; shard_idx < shard_count; ++shard_idx) {
    score_promises.emplace_back(thread_pool.Launch(
        [this, C_pos, C_neg, max_iter, shard_idx, shard_count]() {
          if (params_.mode == kModeRegression) {
            return SolveSVR(C_pos, max_iter, shard_idx, shard_count);
          } else {
            return SolveSVC(C_pos, C_neg, max_iter, shard_idx, shard_count);
          }
        }));
  }

  std::vector<float> scores;
  float sum_score = 0.0f, min_score = HUGE_VAL, max_score = -HUGE_VAL;
  for (auto& v : score_promises) {
    auto score = v.get();
    sum_score += score;
    scores.emplace_back(score);

    min_score = std::min(score, min_score);
    max_score = std::max(score, max_score);
  }

  const auto avg_score = sum_score / shard_count;

  const auto new_best = (params_.cost_function == kCostFunctionRMSE)
                            ? (avg_score < optimize_best_score_)
                            : (avg_score > optimize_best_score_);

  if (params_.do_debug) {
    std::string score_type;
    switch (params_.cost_function) {
      case kCostFunctionAUROC:
        score_type = "AUROC";
        break;
      case kCostFunctionRMSE:
        score_type = "RMSE";
        break;
      case kCostFunctionFn:
        if (!params_.cost_parameter)
          score_type = "precision";
        else
          score_type = ev::cat("F", ev::DoubleToString(params_.cost_parameter));
        break;
      default:
        KJ_FAIL_REQUIRE("Unknown cost function");
    }

    if (new_best) fprintf(stderr, "\033[32;1m");
    fprintf(stderr, "mean_%1$s=%2$.4f min_%1$s=%3$.4f max_%1$s=%4$.4f\n",
            score_type.c_str(), avg_score, min_score, max_score);
    if (new_best) fprintf(stderr, "\033[m");
  }

  if (new_best) {
    optimize_best_score_ = avg_score;
    optimize_minimum_ = cache_key;
  }

  optimize_cache_[cache_key] = avg_score;

  return std::make_pair(avg_score, new_best);
}

float TextClassifierSVMModel::SolveSVC(float C_pos, float C_neg,
                                       size_t max_iter, size_t shard_idx,
                                       size_t shard_count,
                                       std::vector<float>* result_w,
                                       HashWeights* result_weights) {
  const float diag[3] = {0.5f / C_neg, 0.0f, 0.05f / C_pos};

  std::vector<float> alpha;
  if (shard_count > 1) alpha = std::move(shard_alpha_[shard_idx]);
  alpha.resize(documents_.size(), 0.0f);

  std::vector<float> w(index_count_, 0.0f);

  std::vector<size_t> index;
  index.reserve(documents_.size());

  std::vector<float> qd(documents_.size(), 0.0f);

  std::vector<size_t> test_set;

  if (shard_count > 1) {
    for (size_t i = 0; i < document_order_.size(); ++i) {
      const auto document_idx = document_order_[i];

      size_t document_shard;
      if (i < class_split_) {
        document_shard = i * shard_count / class_split_;
      } else {
        document_shard = (i - class_split_) * shard_count /
                         (document_order_.size() - class_split_);
      }

      if (document_shard == shard_idx) {
        test_set.emplace_back(document_idx);
        // Verify that we haven't trained on this data before.
        KJ_ASSERT(!alpha[document_idx], alpha[document_idx]);
      } else {
        index.emplace_back(document_idx);
      }
    }

    KJ_ASSERT(test_set.size() > 0);
  } else {
    index = document_order_;
  }

  KJ_ASSERT(index.size() > 0);

  HashWeights feature_weights;
  std::vector<float> document_scales;

  if (shard_count > 1 && !shard_feature_weights_[shard_idx].empty()) {
    feature_weights = std::move(shard_feature_weights_[shard_idx]);
  } else {
    feature_weights = GetHashWeights(index);
  }

  if (params_.do_normalize) {
    for (const auto& document : documents_) {
      auto sum_sq = Pow2(params_.intercept);
      if (doc_hash_counts_.empty()) {
        for (size_t j = 0; j < document.hash_count; ++j) {
          const auto hash_idx = doc_hash_idx_[document.first_idx + j];
          sum_sq += Pow2(std::get<float>(feature_weights[hash_idx]));
        }
      } else {
        for (size_t j = 0; j < document.hash_count; ++j) {
          const auto hash_idx = doc_hash_idx_[document.first_idx + j];
          if (doc_hash_counts_[document.first_idx + j] <
              std::get<HashCountType>(feature_weights[hash_idx]))
            continue;
          sum_sq += Pow2(std::get<float>(feature_weights[hash_idx]));
        }
      }

      document_scales.emplace_back(sum_sq ? 1.0f / std::sqrt(sum_sq) : 0.0f);
    }
  } else {
    document_scales.resize(documents_.size(), 1.0f);
  }

  for (const auto document_idx : index) {
    const auto& document = documents_[document_idx];

    qd[document_idx] = diag[y_binary_[document_idx] + 1];

    const auto scale = document_scales[document_idx];

    if (doc_hash_counts_.empty()) {
      for (size_t j = 0; j < document.hash_count; ++j) {
        const auto hash_idx = doc_hash_idx_[document.first_idx + j];
        auto val = std::get<float>(feature_weights[hash_idx]) * scale;
        w[hash_idx] += alpha[document_idx] * y_binary_[document_idx] * val;
        qd[document_idx] += Pow2(val);
      }
    } else {
      for (size_t j = 0; j < document.hash_count; ++j) {
        const auto hash_idx = doc_hash_idx_[document.first_idx + j];
        if (doc_hash_counts_[document.first_idx + j] <
            std::get<HashCountType>(feature_weights[hash_idx]))
          continue;
        auto val = std::get<float>(feature_weights[hash_idx]) * scale;
        w[hash_idx] += alpha[document_idx] * y_binary_[document_idx] * val;
        qd[document_idx] += Pow2(val);
      }
    }

    auto icpt = params_.intercept * scale;

    w[0] += alpha[document_idx] * y_binary_[document_idx] * icpt;
    qd[document_idx] += Pow2(icpt);
  }

  size_t iter = 0;
  auto active_size = index.size();

  // PG: projected gradient, for shrinking and stopping
  float PGmax_old = HUGE_VAL;
  float PGmin_old = -HUGE_VAL;

  std::mt19937_64 rng(0);

  while (iter < max_iter) {
    float PGmax_new = -HUGE_VAL;
    float PGmin_new = HUGE_VAL;

    std::shuffle(&index[0], &index[active_size], rng);

    for (size_t s = 0; s < active_size; ++s, ++iter) {
      auto i = index[s];
      const auto yi = y_binary_[i];
      const auto G = Dot(i, w, feature_weights) * document_scales[i] * yi -
                     1.0f + alpha[i] * diag[yi + 1];

      auto PG = 0.0f;
      if (alpha[i] == 0) {
        if (G > PGmax_old) {
          active_size--;
          std::swap(index[s], index[active_size]);
          s--;
          continue;
        } else if (G < 0) {
          PG = G;
        }
      } else {
        PG = G;
      }

      PGmax_new = std::max(PGmax_new, PG);
      PGmin_new = std::min(PGmin_new, PG);

      if (std::fabs(PG) > 1.0e-12f) {
        auto alpha_old = alpha[i];
        alpha[i] = std::max(0.0f, alpha[i] - G / qd[i]);

        AddScaled(w, i, (alpha[i] - alpha_old) * yi * document_scales[i],
                  feature_weights);
      }
    }

    if (PGmax_new - PGmin_new <= params_.epsilon) {
      if (active_size == index.size()) break;

      active_size = index.size();
      PGmax_old = HUGE_VAL;
      PGmin_old = -HUGE_VAL;
      continue;
    }

    PGmax_old = PGmax_new;
    PGmin_old = PGmin_new;
    if (PGmax_old <= 0) PGmax_old = HUGE_VAL;
    if (PGmin_old >= 0) PGmin_old = -HUGE_VAL;
  }

  double result = 0.0;

  switch (params_.cost_function) {
    case kCostFunctionAUROC: {
      double auc_roc = 0.0;

      if (shard_count > 1) {
        // Calculate area under ROC using the data that wasn't used for
        // training.
        std::vector<std::pair<float, float>> results;
        size_t negative_count = 0;
        size_t positive_count = 0;

        for (const auto document_idx : test_set) {
          results.emplace_back(Dot(document_idx, w, feature_weights) *
                                   document_scales[document_idx],
                               y_binary_[document_idx]);

          if (y_binary_[document_idx] == -1)
            ++positive_count;
          else
            ++negative_count;
        }

        std::sort(results.begin(), results.end(),
                  [](const auto& lhs, const auto& rhs) {
                    if (lhs.first != rhs.first) return lhs.first < rhs.first;
                    return lhs.second > rhs.second;
                  });

        size_t seen_negative = 0;
        for (const auto& r : results) {
          if (r.second == -1) {
            ++seen_negative;
          } else {
            KJ_ASSERT(r.second == 1);
            auc_roc += static_cast<double>(seen_negative) /
                       (negative_count * positive_count);
          }
        }

        shard_alpha_[shard_idx] = std::move(alpha);
        shard_feature_weights_[shard_idx] = std::move(feature_weights);
      }

      result = auc_roc;
    } break;

    case kCostFunctionFn: {
      size_t true_positives = 0;
      size_t false_positives = 0;
      size_t false_negatives = 0;
      for (const auto document_idx : test_set) {
        const auto score = Dot(document_idx, w, feature_weights);

        if (score > 0) {
          if (y_binary_[document_idx] > 0) {
            ++true_positives;
          } else {
            ++false_positives;
          }
        } else if (y_binary_[document_idx] > 0) {
          ++false_negatives;
        }
      }

      const auto precision = static_cast<double>(true_positives) /
                             (true_positives + false_positives);
      const auto recall = static_cast<double>(true_positives) /
                          (true_positives + false_negatives);

      result = (1.0 + Pow2(params_.cost_parameter)) * precision * recall /
               (Pow2(params_.cost_parameter) * precision + recall);
    } break;

    default:
      KJ_FAIL_REQUIRE("Unsupported cost function", params_.cost_function);
  }

  if (result_w && result_weights) {
    *result_w = std::move(w);
    *result_weights = std::move(feature_weights);
  }

  return result;
}

float TextClassifierSVMModel::SolveSVR(float C, size_t max_iter,
                                       size_t shard_idx, size_t shard_count,
                                       std::vector<float>* result_w,
                                       HashWeights* result_weights) {
  const auto lambda = 0.5 / C;

  std::vector<float> beta(documents_.size(), 0.0);

  std::vector<float> w(index_count_, 0.0f);

  std::vector<size_t> index;
  index.reserve(documents_.size());

  std::vector<float> qd(documents_.size(), 0.0f);

  std::vector<size_t> test_set;

  for (size_t i = 0; i < document_order_.size(); ++i) {
    const auto document_idx = document_order_[i];

    if (shard_count > 1 && (i % shard_count) == shard_idx) {
      test_set.emplace_back(document_idx);
    } else {
      index.emplace_back(document_idx);
    }
  }

  auto feature_weights = GetHashWeights(index);
  std::vector<float> document_scales;

  for (const auto& document : documents_) {
    float sum_sq = Pow2(params_.intercept);
    if (doc_hash_counts_.empty()) {
      for (size_t i = 0; i < document.hash_count; ++i) {
        const auto hash_idx = doc_hash_idx_[document.first_idx + i];
        sum_sq += Pow2(std::get<float>(feature_weights[hash_idx]));
      }
    } else {
      for (size_t i = 0; i < document.hash_count; ++i) {
        const auto hash_idx = doc_hash_idx_[document.first_idx + i];
        if (doc_hash_counts_[document.first_idx + i] <
            std::get<HashCountType>(feature_weights[hash_idx]))
          continue;
        sum_sq += Pow2(std::get<float>(feature_weights[hash_idx]));
      }
    }

    document_scales.emplace_back(sum_sq ? 1.0f / sum_sq : 0.0f);
  }

  for (const auto document_idx : index) {
    const auto& document = documents_[document_idx];

    const auto scale = document_scales[document_idx];

    if (doc_hash_counts_.empty()) {
      for (size_t j = 0; j < document.hash_count; ++j) {
        const auto hash_idx = doc_hash_idx_[document.first_idx + j];
        auto val = std::get<float>(feature_weights[hash_idx]) * scale;
        w[hash_idx] += beta[document_idx] * y_scalar_[document_idx] * val;
        qd[document_idx] += Pow2(val);
      }
    } else {
      for (size_t j = 0; j < document.hash_count; ++j) {
        const auto hash_idx = doc_hash_idx_[document.first_idx + j];
        if (doc_hash_counts_[document.first_idx + j] <
            std::get<HashCountType>(feature_weights[hash_idx]))
          continue;
        auto val = std::get<float>(feature_weights[hash_idx]) * scale;
        w[hash_idx] += beta[document_idx] * y_scalar_[document_idx] * val;
        qd[document_idx] += Pow2(val);
      }
    }

    auto icpt = params_.intercept * scale;

    w[0] += beta[document_idx] * y_scalar_[document_idx] * icpt;
    qd[document_idx] += icpt * icpt;
  }

  size_t iter = 0;
  auto active_size = index.size();

  float Gmax_old = HUGE_VAL;
  float Gnorm1_init = -1.0;

  std::mt19937_64 rng(0);

  for (; iter < max_iter; ++iter) {
    auto Gmax_new = 0.0f;
    auto Gnorm1_new = 0.0f;

    std::shuffle(&index[0], &index[active_size], rng);

    for (size_t s = 0; s < active_size; ++s) {
      auto i = index[s];
      const auto yi = y_scalar_[i];
      const auto G = Dot(i, w, feature_weights) * document_scales[i] - yi +
                     lambda * beta[i];
      const auto H = qd[i] + lambda;

      const auto Gp = G + params_.regression_epsilon;
      const auto Gn = G - params_.regression_epsilon;
      auto violation = 0.0f;

      if (beta[i] == 0) {
        if (Gp < 0) {
          violation = -Gp;
        } else if (Gn > 0) {
          violation = Gn;
        } else if (Gp > Gmax_old && Gn < -Gmax_old) {
          active_size--;
          std::swap(index[s], index[active_size]);
          s--;
          continue;
        }
      } else if (beta[i] > 0) {
        violation = std::fabs(Gp);
      } else {
        violation = std::fabs(Gn);
      }

      Gmax_new = std::max(Gmax_new, violation);
      Gnorm1_new += violation;

      float d;
      if (Gp < H * beta[i]) {
        d = -Gp / H;
      } else if (Gn > H * beta[i]) {
        d = -Gn / H;
      } else {
        d = -beta[i];
      }

      if (std::fabs(d) > 1.0e-12f) {
        const auto beta_old = beta[i];
        beta[i] += d;

        AddScaled(w, i, (beta[i] - beta_old) * document_scales[i],
                  feature_weights);
      }
    }

    if (iter == 0) Gnorm1_init = Gnorm1_new;

    if (Gnorm1_new <= params_.epsilon * Gnorm1_init) {
      if (active_size == index.size()) break;

      active_size = index.size();
      Gmax_old = HUGE_VAL;
      continue;
    }

    Gmax_old = Gmax_new;
  }

  if (iter >= max_iter) {
    fprintf(stderr, "*** Reached maximum iteration count (%zu/%zu)\n", iter,
            max_iter);
  }

  KJ_REQUIRE(params_.cost_function == kCostFunctionRMSE);

  double rmse = 0.0;

  if (shard_count > 1) {
    double sum_sqerr = 0.0;

    for (const auto i : test_set) {
      const auto d = std::max(
          0.0f, std::fabs(Dot(i, w, feature_weights) * document_scales[i] -
                          y_scalar_[i]) -
                    params_.regression_epsilon);
      sum_sqerr += d * d;
    }

    rmse = std::sqrt(sum_sqerr / test_set.size());
  }

  if (result_w && result_weights) {
    *result_w = std::move(w);
    *result_weights = std::move(feature_weights);
  }

  return rmse;
}

REGISTER_TEXT_CLASSIFIER_MODEL(TextClassifierSVMModel, "l-svm");

}  // namespace ev
