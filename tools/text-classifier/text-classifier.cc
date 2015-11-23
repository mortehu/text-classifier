#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <unordered_map>

#include <err.h>
#include <getopt.h>
#include <sysexits.h>

#include <kj/debug.h>
#include <sparsehash/dense_hash_map>

#include "base/cat.h"
#include "base/columnfile.h"
#include "base/error.h"
#include "base/file.h"
#include "base/hash.h"
#include "base/macros.h"
#include "base/parallel-sort.h"
#include "base/random.h"
#include "base/string.h"
#include "base/stringref.h"
#include "base/thread-pool.h"
#include "tools/text-classifier/23andme.h"
#include "tools/text-classifier/html-tokenizer.h"

using Clock = std::chrono::steady_clock;

namespace {

typedef uint8_t HashCountType;

struct Header {
  float class_id;
  uint32_t hash_count;
};

enum Strategy {
  kStrategy23AndMe,
  kStrategyHTML,
  kStrategyPlainText,
  kStrategySubstrings,
};

enum WeightType { kWeightBNS, kWeightIDF, kWeightLogOdds, kWeightNone };

enum Option : char {
  kOptionC = 'C',
  kOptionClassThreshold = 'c',
  kOptionEpsilon = 'e',
  kOptionIntercept = 'i',
  kOptionMaxIter = 'I',
  kOptionMinC = 'M',
  kOptionMinCount = 'm',
  kOptionRegressionEpsilon = 'r',
  kOptionStrategy = 's',
  kOptionThreshold = 't',
  kOptionWeight = 'w',
  kOptionWeightRatio = 'R'
};

int print_version = 0;
int print_help = 0;

// Shuffle documents before dividing into folds for cross-validation.
int do_shuffle = 1;

// Set to 1 to do regression instead of classification.
int do_regression = 0;

// Set to 0 to use feature counts, not just presence.
int do_unique = 1;

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

// Set to 1 to print debug information.
int do_debug;

uint64_t threshold = 2;

int do_normalize = 1;
Strategy strategy = kStrategyHTML;
WeightType weight_type = kWeightLogOdds;

struct option kLongOptions[] = {
    {"C", required_argument, nullptr, kOptionC},
    {"class-threshold", required_argument, nullptr, kOptionClassThreshold},
    {"no-debug", no_argument, &do_debug, 0},
    {"epsilon", required_argument, nullptr, kOptionEpsilon},
    {"intercept", required_argument, nullptr, kOptionIntercept},
    {"max-iter", required_argument, nullptr, kOptionMaxIter},
    {"min-c", required_argument, nullptr, kOptionMinC},
    {"min-count", required_argument, nullptr, kOptionMinCount},
    {"no-normalize", no_argument, &do_normalize, 0},
    {"no-shuffle", no_argument, &do_shuffle, 0},
    {"no-unique", no_argument, &do_unique, 0},
    {"regression", no_argument, &do_regression, 1},
    {"strategy", required_argument, nullptr, kOptionStrategy},
    {"threshold", required_argument, nullptr, kOptionThreshold},
    {"weight", required_argument, nullptr, kOptionWeight},
    {"weight-ratio", required_argument, nullptr, kOptionWeightRatio},
    {"version", no_argument, &print_version, 1},
    {"help", no_argument, &print_help, 1},
    {nullptr, 0, nullptr, 0}};

template <typename T>
T Pow2(T v) {
  return v * v;
}

void TokenizeSubstrings(const char* begin, const char* end,
                        std::vector<uint64_t>& result) {
  for (auto start = begin; start + 3 <= end; ++start) {
    for (auto stop = start + 3; stop <= end; ++stop) {
      result.emplace_back(ev::Hash(ev::StringRef(start, stop)));
    }
  }
}

void TokenizePlainText(const char* begin, const char* end,
                       std::vector<uint64_t>& result) {
  static const std::array<uint64_t, 3> kWindowMultipliers{{3, 9, 17}};

  std::deque<uint64_t> window;

  auto token_begin = begin;

  while (token_begin != end) {
    if (std::isspace(*token_begin)) {
      ++token_begin;
      continue;
    }

    auto token_end = token_begin + 1;

    if (std::isalnum(*token_begin)) {
      while (token_end != end &&
             (static_cast<uint8_t>(*token_end) >= 0x80 ||
              std::isalnum(*token_end) || *token_end == '\''))
        ++token_end;
    } else {
      while (token_end != end && static_cast<uint8_t>(*token_end) < 0x80 &&
             !std::isalnum(*token_end) && !std::isspace(*token_end))
        ++token_end;
    }

    const ev::StringRef token(token_begin, token_end - token_begin);

    token_begin = token_end;

    const auto token_hash = ev::Hash(token);

    result.emplace_back(token_hash);

    for (size_t i = 0; i < window.size(); ++i)
      result.emplace_back(token_hash + kWindowMultipliers[i] * window[i]);

    window.emplace_front(token_hash);
    if (window.size() > kWindowMultipliers.size()) window.pop_back();
  }
}

void Tokenize(const char* begin, const char* end,
              std::vector<uint64_t>& result) {
  switch (strategy) {
    case kStrategy23AndMe:
      ev::Tokenize23AndMe(begin, end, result);
      break;

    case kStrategyHTML: {
      ev::HTMLTokenizer tok;
      tok.Tokenize(begin, end, result);
    } break;

    case kStrategyPlainText:
      TokenizePlainText(begin, end, result);
      break;

    case kStrategySubstrings:
      TokenizeSubstrings(begin, end, result);
      break;
  }
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

class SVMSolver {
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

  SVMSolver(kj::AutoCloseFd input) {
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

    ev::ColumnFileReader reader(std::move(input));

    while (!reader.End()) {
      auto row = reader.GetRow();
      KJ_REQUIRE(row.size() == 2);
      KJ_REQUIRE(row[0].first == 0, row[0].first);
      KJ_REQUIRE(row[1].first == 1, row[1].first);
      KJ_REQUIRE(row[0].second.size() == sizeof(Header), row[0].second.size());
      KJ_REQUIRE((row[1].second.size() % sizeof(uint64_t)) == 0,
                 row[1].second.size());

      const auto& header =
          *reinterpret_cast<const Header*>(row[0].second.data());

      if (do_regression) y_scalar_.emplace_back(header.class_id);

      if (header.class_id <= class_threshold) {
        if (!do_regression) y_binary_.emplace_back(-1);
        ++n_negative_;
      } else {
        if (!do_regression) y_binary_.emplace_back(1);
        ++n_positive_;
      }

      auto hashes = reinterpret_cast<const uint64_t*>(row[1].second.data());

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
          if (common_hashes[cache_key].second < threshold) {
            common_hashes[cache_key].first = hash;
            common_hashes[cache_key].second = 1;
          }
        } else if (common_hashes[cache_key].second < threshold) {
          all_hashes.emplace_back(hash);
          ++common_hashes[cache_key].second;
        }
      }

      ++document_count;
    }

    common_hashes.clear();
    common_hashes.shrink_to_fit();

    ev::ThreadPool thread_pool;
    ev::ParallelSort(all_hashes.begin(), all_hashes.end(),
                     std::less<uint64_t>(), thread_pool);

    size_t passing_features = 0, total_features = 0;

    for (size_t i = 0; i < all_hashes.size(); ++i) {
      if ((i + 1) == all_hashes.size() || all_hashes[i + 1] != all_hashes[i]) {
        ++total_features;
        if (i + 1 >= threshold &&
            all_hashes[i] == all_hashes[i - threshold + 1])
          all_hashes[passing_features++] = all_hashes[i];
      }
    }

    if (do_debug) {
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

    if (do_regression) {
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
      mean_ = sum / y_scalar_.size();
      for (auto& y : y_scalar_) y -= mean_;
    }

    index_count_ = hash_to_idx_.size() + 1;

    reader.SeekToStart();

    while (!reader.End()) {
      auto row = reader.GetRow();

      const auto& header =
          *reinterpret_cast<const Header*>(row[0].second.data());

      Document doc;
      doc.first_idx = doc_hash_idx_.size();
      doc.hash_count = header.hash_count;
      doc.class_id = header.class_id;

      uint64_t last_hash = 0;

      auto hashes = reinterpret_cast<const uint64_t*>(row[1].second.data());

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
          if (doc_hash_counts_.back() <
              std::numeric_limits<HashCountType>::max())
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
    if (!doc_hash_counts_.empty())
      doc_hash_counts_.resize(doc_hash_idx_.size());

    if (param_c) {
      C_min_ = param_c;
      C_max_ = param_c;
    } else {
      uint32_t max_count = 0;
      for (const auto& document : documents_)
        max_count = std::max(max_count, document.hash_count);
      C_min_ = 1.0f / (2.0f * documents_.size() * max_count);
      // Round down to nearest power of two.
      C_min_ = std::floor(std::log(C_min_) / std::log(2.0f));
      // Clamp to a reasonable lower bound.
      C_min_ = std::max(do_normalize ? -10.0f : -16.0f, C_min_);

      C_min_ = std::max(param_min_c, std::pow(2.0f, C_min_));
    }
  }

  void Solve(float eps) {
    size_t shard_count = 10;
    if (!do_regression) {
      shard_count = std::max(
          2_z, std::min(n_positive_ / 2, std::min(n_negative_ / 2, 10_z)));
    }

    document_order_.clear();

    for (size_t i = 0; i < documents_.size(); ++i)
      document_order_.emplace_back(i);

    if (do_shuffle) {
      std::shuffle(document_order_.begin(), document_order_.end(),
                   std::mt19937_64(1234));
    }

    auto partition_point = std::stable_partition(
        document_order_.begin(), document_order_.end(), [this](const auto idx) {
          return documents_[idx].class_id <= class_threshold;
        });
    class_split_ = partition_point - document_order_.begin();

    if (C_min_ != C_max_) {
      optimize_minimum_score_ = do_regression ? HUGE_VAL : 0.0f;

      float C_pos, C_neg;

      if (do_regression) {
        C_pos = C_neg = C_min_;
      } else if (!weight_ratio) {
        C_pos = 2.0f * C_min_ * n_negative_ / (n_positive_ + n_negative_);
        C_neg = 2.0f * C_min_ * n_positive_ / (n_positive_ + n_negative_);
      } else {
        C_pos = C_min_;
        C_neg = C_min_ / weight_ratio;
      }

      // The sequence in which we try scaling `C_pos` and `C_neg`.
      static const std::array<std::pair<float, float>, 5> kGrowthFactors{
          {{2.0f, 2.0f},
           {1.0f, 0.5f},
           {0.5f, 1.0f},
           {1.0f, 2.0f},
           {2.0f, 1.0f}}};

      size_t growth_factor_idx = 0;

      // Set to true if we've seen an improvement in this round, so we should
      // retry the scaling factors from the start before giving up.
      bool do_reset = false;

      for (;;) {
        if (TestParameters(C_pos, C_neg, shard_count, max_iterations, eps)
                .second) {
          if (growth_factor_idx > 0) do_reset = true;
        } else {
          if (do_regression) break;

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
    } else if (weight_ratio) {
      optimize_minimum_.first = C_min_;
      optimize_minimum_.second = C_min_ / weight_ratio;
    } else {
      optimize_minimum_.first =
          2.0f * C_min_ * n_negative_ / (n_positive_ + n_negative_);
      optimize_minimum_.second =
          2.0f * C_min_ * n_positive_ / (n_positive_ + n_negative_);
    }

    if (do_regression) {
      SolveSVR(optimize_minimum_.first, eps, max_iterations, 0, 1, &w_,
               &feature_weights_);
    } else {
      SolveSVC(optimize_minimum_.first, optimize_minimum_.second, eps,
               max_iterations, 0, 1, &w_, &feature_weights_);
    }
  }

  void Save(kj::AutoCloseFd output) const {
    static const auto kBufferLimit = 1024 * 1024;
    std::string buffer;

    auto buffer_append = [&buffer](auto v) {
      buffer.append(reinterpret_cast<const char*>(&v), sizeof(v));
    };

    uint64_t plane_size = (intercept != 0) ? 1 : 0;
    uint64_t feature_weight_count = 0;

    KJ_ASSERT(w_[0] == 0 || intercept != 0, w_[0], intercept);

    for (size_t i = 1; i < w_.size(); ++i)
      if (w_[i]) ++plane_size;

    for (const auto& w : feature_weights_)
      if (std::get<float>(w)) ++feature_weight_count;

    buffer_append(plane_size);
    buffer_append(feature_weight_count);
    buffer_append(mean_);

    size_t plane_weights_written = 0;

    if (intercept != 0) {
      buffer_append(uint64_t(0));
      buffer_append(w_[0]);
      ++plane_weights_written;
    }

    for (const auto& p : hash_to_idx_) {
      const auto weight = w_[p.second];

      if (!weight) continue;

      KJ_ASSERT(std::isfinite(weight), weight);

      buffer_append(p.first);
      buffer_append(weight);

      if (buffer.size() >= kBufferLimit) {
        ev::WriteAll(output, buffer);
        buffer.clear();
      }

      ++plane_weights_written;
    }

    KJ_ASSERT(plane_weights_written == plane_size, plane_size,
              plane_weights_written);

    size_t feature_weights_written = 0;

    for (const auto& p : hash_to_idx_) {
      const auto weight = std::get<float>(feature_weights_[p.second]);

      if (!weight) continue;

      const auto hash_count_threshold =
          std::get<HashCountType>(feature_weights_[p.second]);

      KJ_ASSERT(std::isfinite(weight), weight);

      buffer_append(p.first);
      buffer_append(weight);
      buffer_append(hash_count_threshold);

      if (buffer.size() >= kBufferLimit) {
        ev::WriteAll(output, buffer);
        buffer.clear();
      }

      ++feature_weights_written;
    }

    KJ_ASSERT(feature_weights_written == feature_weight_count,
              feature_weights_written, feature_weight_count);

    ev::WriteAll(output, buffer);
  }

 private:
  typedef std::vector<std::tuple<float, HashCountType>> HashWeights;

  // Returns scaling weights and count thresholds for each feature.
  //
  // See "BNS Feature Scaling: An Improved Representation over TF·IDF for SVM
  // Text Classification" by George Forman.
  HashWeights GetHashWeights(const std::vector<size_t>& training_set) {
    size_t n_positive = 0, n_negative = 0;
    for (const auto document_idx : training_set)
      ++((documents_[document_idx].class_id > class_threshold) ? n_positive
                                                               : n_negative);

    HashWeights result;
    result.reserve(index_count_);

    // Maximum and minimum values for `tpr` and `fpr`, used in BNS calculation.
    const auto pr_min = std::min(0.0005, 1.0 / (n_positive + n_negative));
    const auto pr_max = 1.0 - pr_min;

    const auto get_weight = [n_positive, n_negative, pr_min, pr_max](
        size_t true_positives, size_t false_positives) {
      if (true_positives + false_positives < threshold) return 0.0f;

      const auto tp = static_cast<double>(true_positives);
      const auto fp = static_cast<double>(false_positives);
      const auto fn = n_positive - tp;
      const auto tn = n_negative - fp;
      const auto tpr = std::max(pr_min, std::min(pr_max, tp / n_positive));
      const auto fpr = std::max(pr_min, std::min(pr_max, fp / n_negative));

      float weight;

      switch (weight_type) {
        case kWeightBNS:
          weight = InvNormalCDF(tpr) - InvNormalCDF(fpr);
          break;

        case kWeightIDF:
          weight = std::log((n_positive + n_negative) / (tp + fp));
          break;

        case kWeightLogOdds: {
          const auto num = std::max(tp, min_count) * std::max(tn, min_count);
          const auto den = std::max(fp, min_count) * std::max(fn, min_count);

          weight = std::log(num / den);
        } break;

        case kWeightNone:
          weight = 1.0;
          break;

        default:
          KJ_FAIL_ASSERT("Unknown weight_type", weight_type);
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

        auto label = document.class_id > class_threshold;

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

          if (document.class_id <= class_threshold) {
            ++hash_stats[hash_idx].false_positives;
          } else {
            ++hash_stats[hash_idx].true_positives;
          }
        }
      }

      for (const auto& stat : hash_stats) {
        result.emplace_back(
            get_weight(stat.true_positives, stat.false_positives), 0);
      }
    }

    return result;
  }

  // TODO(mortehu): Use templates to generate multiple versions of `Dot` and
  // `AddScaled`.

  float Dot(size_t idx, const std::vector<float>& w,
            const HashWeights& feature_weights) {
    float result = w[0] * intercept;

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

  void AddScaled(std::vector<float>& w, size_t idx, float scale,
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

    w[0] += scale * intercept;
  }

  std::pair<float, bool> TestParameters(float C_pos, float C_neg,
                                        size_t shard_count, size_t max_iter,
                                        float eps) {
    if (C_pos <= 0 || C_neg <= 0) return std::make_pair(HUGE_VAL, false);

    const auto cache_key = std::make_pair(C_pos, C_neg);

    auto cache_i = optimize_cache_.find(cache_key);
    if (cache_i != optimize_cache_.end())
      return std::make_pair(cache_i->second, false);

    if (do_debug) fprintf(stderr, "C_pos=%.9g C_neg=%.9g ", C_pos, C_neg);

    ev::ThreadPool thread_pool;

    std::vector<std::future<float>> score_promises;

    for (size_t shard_idx = 0; shard_idx < shard_count; ++shard_idx) {
      score_promises.emplace_back(thread_pool.Launch(
          [this, C_pos, C_neg, eps, max_iter, shard_idx, shard_count]() {
            if (do_regression) {
              return SolveSVR(C_pos, eps, max_iter, shard_idx, shard_count);
            } else {
              return SolveSVC(C_pos, C_neg, eps, max_iter, shard_idx,
                              shard_count);
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

    const auto score_type = do_regression ? "RMSE" : "AUROC";

    const auto new_best = do_regression ? (avg_score < optimize_minimum_score_)
                                        : (avg_score > optimize_minimum_score_);

    if (do_debug) {
      if (new_best) fprintf(stderr, "\033[32;1m");
      fprintf(stderr, "mean_%1$s=%2$.4f min_%1$s=%3$.4f max_%1$s=%4$.4f\n",
              score_type, avg_score, min_score, max_score);
      if (new_best) fprintf(stderr, "\033[m");
    }

    if (new_best) {
      optimize_minimum_score_ = avg_score;
      optimize_minimum_ = cache_key;
    }

    optimize_cache_[cache_key] = avg_score;

    return std::make_pair(avg_score, new_best);
  }

  float SolveSVC(float C_pos, float C_neg, float eps, size_t max_iter,
                 size_t shard_idx, size_t shard_count,
                 std::vector<float>* result_w = nullptr,
                 HashWeights* result_weights = nullptr) {
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

    if (do_normalize) {
      for (const auto& document : documents_) {
        auto sum_sq = intercept * intercept;
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

      auto icpt = intercept * scale;

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

      if (PGmax_new - PGmin_new <= eps) {
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

    double auc_roc = 0.0;

    if (shard_count > 1) {
      // Calculate area under ROC using the data that wasn't used for training.
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

    if (result_w && result_weights) {
      *result_w = std::move(w);
      *result_weights = std::move(feature_weights);
    }

    return auc_roc;
  }

  float SolveSVR(float C, float eps, size_t max_iter, size_t shard_idx,
                 size_t shard_count, std::vector<float>* result_w = nullptr,
                 HashWeights* result_weights = nullptr) {
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
      float sum_sq = Pow2(intercept);
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

      auto icpt = intercept * scale;

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

        const auto Gp = G + regression_epsilon;
        const auto Gn = G - regression_epsilon;
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

      if (Gnorm1_new <= eps * Gnorm1_init) {
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

    double rmse = 0.0;

    if (shard_count > 1) {
      double sum_sqerr = 0.0;

      for (const auto i : test_set) {
        const auto d = std::max(
            0.0f, std::fabs(Dot(i, w, feature_weights) * document_scales[i] -
                            y_scalar_[i]) -
                      regression_epsilon);
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

  google::dense_hash_map<uint64_t, uint32_t> hash_to_idx_;

  std::map<std::pair<double, double>, double> optimize_cache_;

  std::pair<double, double> optimize_minimum_{0.0, 0.0};
  float optimize_minimum_score_ = HUGE_VAL;

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

  float mean_ = 0.0f;

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
};

class SVMModel {
 public:
  SVMModel(FILE* model_input) {
    uint64_t decision_plane_size = 0;
    uint64_t feature_weight_count = 0;

    KJ_REQUIRE(1 == fread(&decision_plane_size, sizeof(decision_plane_size), 1,
                          model_input));
    KJ_REQUIRE(1 == fread(&feature_weight_count, sizeof(feature_weight_count),
                          1, model_input));
    KJ_REQUIRE(1 == fread(&bias_, sizeof(bias_), 1, model_input));

    decision_plane_.set_empty_key(0);
    decision_plane_.resize(decision_plane_size);

    for (uint64_t i = 0; i < decision_plane_size; ++i) {
      uint64_t hash = 0;
      float weight = 0.0f;

      KJ_REQUIRE(1 == fread(&hash, sizeof(hash), 1, model_input));
      KJ_REQUIRE(1 == fread(&weight, sizeof(weight), 1, model_input));

      if (!hash) {
        intercept_weight_ = weight;
      } else {
        decision_plane_[hash] = weight;
      }
    }

    classify_feature_weights_.set_empty_key(0);
    classify_feature_weights_.resize(feature_weight_count);

    for (uint64_t i = 0; i < feature_weight_count; ++i) {
      uint64_t hash = 0;
      float weight = 0.0f;
      HashCountType hash_count_threshold = 0;

      KJ_REQUIRE(1 == fread(&hash, sizeof(hash), 1, model_input));
      KJ_REQUIRE(1 == fread(&weight, sizeof(weight), 1, model_input));
      KJ_REQUIRE(1 == fread(&hash_count_threshold, sizeof(hash_count_threshold),
                            1, model_input));
      KJ_REQUIRE(hash != 0);

      classify_feature_weights_[hash] =
          std::make_pair(weight, hash_count_threshold);
    }
  }

  float Classify(const ev::StringRef& data) {
    std::vector<uint64_t> doc_hashes;
    Tokenize(data.begin(), data.end(), doc_hashes);

    std::sort(doc_hashes.begin(), doc_hashes.end());

    // Make sure zero is not present.
    while (!doc_hashes.empty() && !doc_hashes.front())
      doc_hashes.erase(doc_hashes.begin());

    float v = intercept_weight_;
    auto sum_sq = Pow2(intercept);

    for (auto i = doc_hashes.begin(); i != doc_hashes.end();) {
      const auto hash = *i++;
      KJ_ASSERT(hash > 0);

      HashCountType count = 0;

      while (i != doc_hashes.end() && *i == hash) {
        if (count < std::numeric_limits<HashCountType>::max()) ++count;
        ++i;
      }

      auto w = classify_feature_weights_.find(hash);
      if (w == classify_feature_weights_.end()) continue;

      if (count < std::get<HashCountType>(w->second)) continue;

      sum_sq += Pow2(std::get<float>(w->second));

      auto j = decision_plane_.find(hash);
      if (j != decision_plane_.end())
        v += j->second * std::get<float>(w->second);
    }

    if (do_normalize && sum_sq) v /= std::sqrt(sum_sq);

    return v + bias_;
  }

 private:
  // Model used during classification.
  float intercept_weight_ = 0.0f;
  float bias_ = 0.0f;
  google::dense_hash_map<uint64_t, float> decision_plane_;
  google::dense_hash_map<uint64_t, std::pair<float, HashCountType>>
      classify_feature_weights_;
};

}  // namespace

int main(int argc, char** argv) try {
  do_debug = isatty(STDERR_FILENO);

  int i;
  while ((i = getopt_long(argc, argv, "C:", kLongOptions, 0)) != -1) {
    if (!i) continue;
    if (i == '?')
      errx(EX_USAGE, "Try '%s --help' for more information.", argv[0]);

    switch (static_cast<Option>(i)) {
      case kOptionC:
        param_c = ev::StringToDouble(optarg);
        break;

      case kOptionClassThreshold:
        class_threshold = ev::StringToDouble(optarg);
        break;

      case kOptionEpsilon:
        epsilon = ev::StringToDouble(optarg);
        break;

      case kOptionIntercept:
        intercept = ev::StringToDouble(optarg);
        break;

      case kOptionMaxIter:
        max_iterations = ev::StringToUInt64(optarg);
        break;

      case kOptionMinC:
        param_min_c = ev::StringToDouble(optarg);
        break;

      case kOptionMinCount:
        min_count = ev::StringToDouble(optarg);
        break;

      case kOptionRegressionEpsilon:
        regression_epsilon = ev::StringToDouble(optarg);
        break;

      case kOptionStrategy:
        if (!strcmp(optarg, "23andme")) {
          strategy = kStrategy23AndMe;
        } else if (!strcmp(optarg, "html")) {
          strategy = kStrategyHTML;
        } else if (!strcmp(optarg, "plain")) {
          strategy = kStrategyPlainText;
        } else if (!strcmp(optarg, "substrings")) {
          strategy = kStrategySubstrings;
        } else {
          KJ_FAIL_REQUIRE("Unknown strategy", optarg);
        }
        break;

      case kOptionThreshold:
        threshold = ev::StringToUInt64(optarg);
        break;

      case kOptionWeight:
        if (!strcmp(optarg, "bns")) {
          weight_type = kWeightBNS;
        } else if (!strcmp(optarg, "idf")) {
          weight_type = kWeightIDF;
        } else if (!strcmp(optarg, "log-odds")) {
          weight_type = kWeightLogOdds;
        } else if (!strcmp(optarg, "none")) {
          weight_type = kWeightNone;
        } else {
          KJ_FAIL_REQUIRE("Unknown weight type", optarg);
        }
        break;

      case kOptionWeightRatio:
        weight_ratio = ev::StringToDouble(optarg);
        break;
    }
  }

  if (print_help) {
    printf(
        "Usage: %s [OPTION]... COMMAND [COMMAND-ARGS]...\n"
        "\n"
        "  --C=VALUE              set C for positive training examples.\n"
        "                         Automatically determined by default\n"
        "  --min-c=VALUE          set minimum C for positive training "
        "examples\n"
        "  --weight-ratio=VALUE   set ratio of C for positive and "
        "negative\n"
        "                         examples.  Automatically determined by "
        "default\n"
        "  --epsilon=EPS          optimization completion threshold [%s]\n"
        "  --strategy=plain|html|substrings|23andme\n"
        "                         select tokenization strategy\n"
        "  --no-normalize         skip L2-normalization\n"
        "  --no-shuffle           skip shuffle before cross-validation\n"
        "  --no-unique            use feature counts\n"
        "\n"
        "  --weight=bns|idf|log-odds|none\n"
        "                         set scaling method\n"
        "  --min-c=VALUE          starting point for C parameter search\n"
        "  --min-count=VALUE      value to substitute for zero when "
        "counting feature\n"
        "                             occurrences during weighting [%s]\n"
        "\n"
        "Regression options:\n"
        "  --regression           do regression instead of classification\n"
        "  --regression-epsilon=EPS\n"
        "                         error tolerance for regression [%s]\n"
        "\n"
        "  --help     display this help and exit\n"
        "  --version  display version information and exit\n"
        "\n"
        "Commands:\n"
        "  learn DB-PATH CLASS-ID - learn the class of a single document\n"
        "  batch-learn - learn the class of any number of documents\n"
        "  analyze DB-PATH MODEL-PATH - build a model based on learned "
        "data\n"
        "  batch-classify - classify any number of documents\n"
        "\n"
        "Tokenization strategies:\n"
        "  plain - make features from skip-grams in plain text\n"
        "  html - like plain, but with added special features for HTML "
        "tags\n"
        "  substrings - make a features from all substrings.  Use only on "
        "very\n"
        "               short inputs\n"
        "  23andme - make features from alleles stored in 23andMe raw data "
        "dump\n"
        "\n"
        "Report bugs to <morten.hustveit@gmail.com>\n",
        argv[0], ev::FloatToString(epsilon).c_str(),
        ev::FloatToString(min_count).c_str(),
        ev::FloatToString(regression_epsilon).c_str());

    return EXIT_SUCCESS;
  }

  if (print_version) {
    puts(PACKAGE_STRING);

    return EXIT_SUCCESS;
  }

  if (optind == argc)
    errx(EX_USAGE, "Usage: %s [OPTION]... [--] COMMAND [COMMAND-ARGS]...",
         argv[0]);

  ev::StringRef command(argv[optind++]);

  if (command == "learn") {
    if (optind + 2 > argc) {
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] learn DB-PATH CLASS-ID",
           argv[0]);
    }

    ev::ColumnFileWriter output(
        ev::OpenFile(argv[optind++], O_WRONLY | O_APPEND | O_CREAT, 0666));

    const auto class_id = ev::StringToFloat(argv[optind++]);

    const bool use_stdin = (optind == argc);
    bool done = false;

    std::vector<std::pair<uint32_t, ev::StringRef>> row;

    while (!done) {
      kj::Array<char> buffer;

      if (use_stdin) {
        buffer = ev::ReadFD(STDIN_FILENO);
        done = true;
      } else {
        buffer = ev::ReadFile(argv[optind++]);
        done = (optind == argc);
      }

      std::vector<uint64_t> doc_hashes;
      Tokenize(buffer.begin(), buffer.end(), doc_hashes);

      std::sort(doc_hashes.begin(), doc_hashes.end());
      if (do_unique) {
        doc_hashes.erase(std::unique(doc_hashes.begin(), doc_hashes.end()),
                         doc_hashes.end());
      }

      // Make sure zero is not present.
      while (!doc_hashes.empty() && !doc_hashes.front())
        doc_hashes.erase(doc_hashes.begin());

      if (doc_hashes.empty()) return EXIT_SUCCESS;

      Header hdr;
      hdr.class_id = class_id;
      hdr.hash_count = doc_hashes.size();

      row.clear();
      row.emplace_back(0, ev::StringRef(reinterpret_cast<const char*>(&hdr),
                                        sizeof(Header)));
      row.emplace_back(
          1, ev::StringRef(reinterpret_cast<const char*>(doc_hashes.data()),
                           sizeof(doc_hashes[0]) * doc_hashes.size()));
      output.PutRow(row);
    }
  } else if (command == "batch-learn") {
    static const auto kFlushInterval = 128_z;
    size_t buffer_fill = 0;

    if (optind + 1 != argc)
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] batch-learn DB-PATH", argv[0]);

    ev::ColumnFileWriter output(
        ev::OpenFile(argv[optind++], O_WRONLY | O_APPEND | O_CREAT, 0666));

    ev::ThreadPool thread_pool;

    float class_id;
    size_t length;

    // Used to make sure documents are output in the same order they were
    // provided.
    auto previous_future = thread_pool.Launch([] { return 0; });

    while (2 == scanf("%g %zu", &class_id, &length)) {
      std::vector<char> buffer;
      KJ_REQUIRE(length > 0, length);

      KJ_REQUIRE('\n' == getchar());

      buffer.resize(length);
      auto ret = fread(buffer.data(), 1, length, stdin);
      KJ_REQUIRE(ret == length, "fread failed");

      previous_future = thread_pool.Launch([
        buffer = std::move(buffer),
        class_id,
        &output,
        &buffer_fill,
        previous_future = std::move(previous_future)
      ]() mutable {
        std::vector<uint64_t> doc_hashes;
        Tokenize(&buffer[0], &buffer[buffer.size()], doc_hashes);

        buffer.clear();
        buffer.shrink_to_fit();

        std::sort(doc_hashes.begin(), doc_hashes.end());
        if (do_unique) {
          doc_hashes.erase(std::unique(doc_hashes.begin(), doc_hashes.end()),
                           doc_hashes.end());
        }

        // Make sure zero is not present.
        while (!doc_hashes.empty() && !doc_hashes.front())
          doc_hashes.erase(doc_hashes.begin());

        // Wait for the previous document to be written.
        previous_future.get();

        if (!doc_hashes.empty()) {
          Header hdr;
          hdr.class_id = class_id;
          hdr.hash_count = doc_hashes.size();

          std::vector<std::pair<uint32_t, ev::StringRef>> row;

          row.emplace_back(0, ev::StringRef(reinterpret_cast<const char*>(&hdr),
                                            sizeof(Header)));
          row.emplace_back(
              1, ev::StringRef(reinterpret_cast<const char*>(doc_hashes.data()),
                               sizeof(doc_hashes[0]) * doc_hashes.size()));
          output.PutRow(row);

          if (++buffer_fill == kFlushInterval) {
            output.Flush();
            buffer_fill = 0;
          }
        }

        return 0;
      });
    }

    previous_future.get();
  } else if (command == "print-model") {
    if (optind + 1 != argc) {
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] print-model MODEL-PATH",
           argv[0]);
    }

    auto model_input = ev::OpenFileStream(argv[optind], "r");

    uint64_t decision_plane_size = 0;
    uint64_t feature_weight_count = 0;
    float bias = 0.0f;

    KJ_REQUIRE(1 == fread(&decision_plane_size, sizeof(decision_plane_size), 1,
                          model_input.get()));
    KJ_REQUIRE(1 == fread(&feature_weight_count, sizeof(feature_weight_count),
                          1, model_input.get()));
    KJ_REQUIRE(1 == fread(&bias, sizeof(bias), 1, model_input.get()));

    printf("bias: %s\n", ev::FloatToString(bias).c_str());

    for (uint64_t i = 0; i < decision_plane_size; ++i) {
      uint64_t hash = 0;
      float weight = 0.0f;

      KJ_REQUIRE(1 == fread(&hash, sizeof(hash), 1, model_input.get()));
      KJ_REQUIRE(1 == fread(&weight, sizeof(weight), 1, model_input.get()));

      if (!hash) {
        printf("intercept: %s\n", ev::FloatToString(weight).c_str());
      } else {
        printf("%" PRIx64 ": %s\n", hash, ev::FloatToString(weight).c_str());
      }
    }

    for (uint64_t i = 0; i < feature_weight_count; ++i) {
      uint64_t hash = 0;
      float weight = 0.0f;
      HashCountType hash_count_threshold = 0;

      KJ_REQUIRE(1 == fread(&hash, sizeof(hash), 1, model_input.get()));
      KJ_REQUIRE(1 == fread(&weight, sizeof(weight), 1, model_input.get()));
      KJ_REQUIRE(1 == fread(&hash_count_threshold, sizeof(hash_count_threshold),
                            1, model_input.get()));
      KJ_REQUIRE(hash != 0);

      printf("%" PRIx64 "_weight: %s\n", hash,
             ev::FloatToString(weight).c_str());
      if (hash_count_threshold != 0)
        printf("%" PRIx64 "_threshold: %u\n", hash, hash_count_threshold);
    }
  } else if (command == "classify") {
    if (optind + 1 > argc) {
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] classify MODEL-PATH [PATH]...",
           argv[0]);
    }

    SVMModel model(ev::OpenFileStream(argv[optind++], "r").get());

    const bool use_stdin = (optind == argc);
    bool done = false;

    while (!done) {
      kj::Array<const char> buffer;

      const char* name = "<stdin>";

      if (use_stdin) {
        buffer = ev::ReadFD(STDIN_FILENO);
        done = true;
      } else {
        name = argv[optind++];
        buffer = ev::ReadFile(name);
        done = (optind == argc);
      }

      auto v = model.Classify(buffer);

      printf("%s\t%.9g\n", name, v);
    }
  } else if (command == "batch-classify") {
    if (optind + 1 != argc)
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] batch-classify MODEL-PATH",
           argv[0]);

    SVMModel model(ev::OpenFileStream(argv[optind], "r").get());

    ev::ThreadPool thread_pool;
    std::mutex output_lock;

    // XXX: Fix buffer overflow bug (reading into `key`).
    char key[256];
    size_t length;

    while (2 == scanf("%s %zu", key, &length)) {
      KJ_REQUIRE(length > 0, length);

      KJ_REQUIRE('\n' == getchar());

      std::vector<char> buffer(length);

      auto ret = fread(buffer.data(), 1, length, stdin);
      KJ_REQUIRE(ret == length, "fread failed");

      thread_pool.Launch([
        buffer = std::move(buffer),
        key = std::string(key),
        &output_lock,
        &model
      ]() mutable {
        auto v = model.Classify(buffer);

        std::unique_lock<std::mutex> lk(output_lock);
        printf("%s\t%.9g\n", key.c_str(), v);
        fflush(stdout);
      });
    }

    thread_pool.Wait();
  } else if (command == "analyze") {
    if (optind + 2 != argc) {
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] analyze DB-PATH MODEL-PATH",
           argv[0]);
    }

    SVMSolver solver(ev::OpenFile(argv[optind++], O_RDONLY));
    solver.Solve(epsilon);
    solver.Save(
        ev::OpenFile(argv[optind++], O_WRONLY | O_CREAT | O_TRUNC, 0666));
  } else {
    KJ_FAIL_REQUIRE("Unknown command", command.str());
  }
} catch (kj::Exception e) {
  KJ_LOG(FATAL, e);
  return EXIT_FAILURE;
}
