#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <map>

#include <err.h>
#include <getopt.h>
#include <sysexits.h>

#include <columnfile.h>
#include <kj/debug.h>
#include <sparsehash/dense_hash_map>

#include "base/error.h"
#include "base/file.h"
#include "base/hash.h"
#include "base/macros.h"
#include "base/random.h"
#include "base/string.h"
#include "base/stringref.h"
#include "base/thread-pool.h"
#include "tools/text-classifier/23andme.h"
#include "tools/text-classifier/common.h"
#include "tools/text-classifier/html-tokenizer.h"
#include "tools/text-classifier/model.h"
#include "tools/text-classifier/utf8.h"

using Clock = std::chrono::steady_clock;

namespace {

enum Strategy {
  kStrategy23AndMe,
  kStrategyHTML,
  kStrategyPlainText,
  kStrategySubstrings,
};

enum Option : char {
  kOptionC = 'C',
  kOptionClassThreshold = 'c',
  kOptionCostFunction = 'f',
  kOptionEpsilon = 'e',
  kOptionIntercept = 'i',
  kOptionMaxIter = 'I',
  kOptionMinC = 'M',
  kOptionMinCount = 'm',
  kOptionModelType = 'T',
  kOptionNoDebug = 'D',
  kOptionNoNormalize = 'N',
  kOptionNoShuffle = 'S',
  kOptionRegressionEpsilon = 'r',
  kOptionStrategy = 's',
  kOptionThreshold = 't',
  kOptionWeight = 'w',
  kOptionWeightRatio = 'R',
};

int print_version = 0;
int print_help = 0;

// Set to 1 for regression, 0 for classification.
int do_regression = 0;

// Set to 0 to use feature counts, not just presence.
int do_unique = 1;

Strategy strategy = kStrategyHTML;

struct option kLongOptions[] = {
    {"C", required_argument, nullptr, kOptionC},
    {"class-threshold", required_argument, nullptr, kOptionClassThreshold},
    {"cost-function", required_argument, nullptr, kOptionCostFunction},
    {"epsilon", required_argument, nullptr, kOptionEpsilon},
    {"intercept", required_argument, nullptr, kOptionIntercept},
    {"max-iter", required_argument, nullptr, kOptionMaxIter},
    {"min-c", required_argument, nullptr, kOptionMinC},
    {"min-count", required_argument, nullptr, kOptionMinCount},
    {"model-type", required_argument, nullptr, kOptionModelType},
    {"no-debug", no_argument, nullptr, kOptionNoDebug},
    {"no-normalize", no_argument, nullptr, kOptionNoNormalize},
    {"no-shuffle", no_argument, nullptr, kOptionNoShuffle},
    {"no-unique", no_argument, &do_unique, 0},
    {"regression", no_argument, &do_regression, 1},
    {"strategy", required_argument, nullptr, kOptionStrategy},
    {"threshold", required_argument, nullptr, kOptionThreshold},
    {"weight", required_argument, nullptr, kOptionWeight},
    {"weight-ratio", required_argument, nullptr, kOptionWeightRatio},
    {"version", no_argument, &print_version, 1},
    {"help", no_argument, &print_help, 1},
    {nullptr, 0, nullptr, 0}};

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

  ev::StringRef data(begin, end);
  while (!data.empty()) {
    const auto ch = GetUtf8Char(&data);
    if (ch >= 0x80) result.emplace_back(ch);
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

}  // namespace

int main(int argc, char** argv) try {
  const auto model_types =
      ev::TextClassifierModelFactory::GetInstance()->GetModelTypes();
  KJ_REQUIRE(model_types.size() > 0);

  auto model_type = model_types[0];

  TextClassifierParams params;

  params.do_debug = isatty(STDERR_FILENO);

  int i;
  while ((i = getopt_long(argc, argv, "C:", kLongOptions, 0)) != -1) {
    if (!i) continue;
    if (i == '?')
      errx(EX_USAGE, "Try '%s --help' for more information.", argv[0]);

    switch (static_cast<Option>(i)) {
      case kOptionC:
        params.param_c = ev::StringToDouble(optarg);
        break;

      case kOptionClassThreshold:
        params.class_threshold = ev::StringToDouble(optarg);
        break;

      case kOptionCostFunction:
        if (!strcmp(optarg, "auroc")) {
          params.cost_function = kCostFunctionAUROC;
        } else if (!strcmp(optarg, "rmse")) {
          params.cost_function = kCostFunctionRMSE;
        } else if (optarg[0] == 'f' && std::isdigit(optarg[1])) {
          params.cost_function = kCostFunctionFn;
          params.cost_parameter = ev::StringToDouble(optarg + 1);
        } else if (!strcmp(optarg, "precision")) {
          // Precision is the same as F0.
          params.cost_function = kCostFunctionFn;
          params.cost_parameter = 0;
        } else {
          KJ_FAIL_REQUIRE("Unknown cost function", optarg);
        }
        break;

      case kOptionEpsilon:
        params.epsilon = ev::StringToDouble(optarg);
        break;

      case kOptionIntercept:
        params.intercept = ev::StringToDouble(optarg);
        break;

      case kOptionNoDebug:
        params.do_debug = false;
        break;

      case kOptionNoNormalize:
        params.do_normalize = false;
        break;

      case kOptionNoShuffle:
        params.do_shuffle = false;
        break;

      case kOptionMaxIter:
        params.max_iterations = ev::StringToUInt64(optarg);
        break;

      case kOptionMinC:
        params.param_min_c = ev::StringToDouble(optarg);
        break;

      case kOptionMinCount:
        params.min_count = ev::StringToDouble(optarg);
        break;

      case kOptionModelType:
        model_type = optarg;
        KJ_REQUIRE(
            model_types.end() !=
                std::find(model_types.begin(), model_types.end(), model_type),
            model_type);
        break;

      case kOptionRegressionEpsilon:
        params.regression_epsilon = ev::StringToDouble(optarg);
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
        params.threshold = ev::StringToUInt64(optarg);
        break;

      case kOptionWeight:
        if (!strcmp(optarg, "bns")) {
          params.weight_type = kWeightBNS;
        } else if (!strcmp(optarg, "idf")) {
          params.weight_type = kWeightIDF;
        } else if (!strcmp(optarg, "log-odds")) {
          params.weight_type = kWeightLogOdds;
        } else if (!strcmp(optarg, "none")) {
          params.weight_type = kWeightNone;
        } else {
          KJ_FAIL_REQUIRE("Unknown weight type", optarg);
        }
        break;

      case kOptionWeightRatio:
        params.weight_ratio = ev::StringToDouble(optarg);
        break;
    }
  }

  if (print_help) {
    printf(
        "Usage: %s [OPTION]... COMMAND [COMMAND-ARGS]...\n"
        "\n"
        "  --model-type=%s\n"
        "                         model learning method\n"
        "  --strategy=plain|html|substrings|23andme\n"
        "                         feature extraction strategy\n"
        "  --no-unique            use feature counts\n"
        "  --regression           do regression instead of classification\n"
        "\n"
        "Linear SVM (l-svm) options:\n"
        "  --epsilon=EPS          optimization completion threshold [%s]\n"
        "  --C=VALUE              set C for positive training examples.\n"
        "                         Automatically determined by default\n"
        "  --min-c=VALUE          set minimum C for positive training "
        "examples\n"
        "  --weight-ratio=VALUE   set ratio of C for positive and "
        "negative\n"
        "                         examples.  Automatically determined by "
        "default\n"
        "  --weight=bns|idf|log-odds|none\n"
        "                         set feature count scaling method\n"
        "  --min-count=VALUE      value to substitute for zero when "
        "counting feature\n"
        "                             occurrences during weighting [%s]\n"
        "\n"
        "Cross-validation options:\n"
        "  --no-normalize         skip L2-normalization\n"
        "  --no-shuffle           skip shuffle before cross-validation\n"
        "\n"
        "  --cost-function=auroc|f1|rmse\n"
        "                        set cost function for hyperparameter "
        "selection\n"
        "\n"
        "Regression options:\n"
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
        argv[0], ev::Join(model_types.begin(), model_types.end(), "|").c_str(),
        ev::FloatToString(params.epsilon).c_str(),
        ev::FloatToString(params.min_count).c_str(),
        ev::FloatToString(params.regression_epsilon).c_str());

    return EXIT_SUCCESS;
  }

  if (print_version) {
    puts(PACKAGE_STRING);

    return EXIT_SUCCESS;
  }

  if (optind == argc)
    errx(EX_USAGE, "Usage: %s [OPTION]... [--] COMMAND [COMMAND-ARGS]...",
         argv[0]);

  if (params.cost_function == kCostFunctionDefault) {
    params.cost_function =
        do_regression ? kCostFunctionRMSE : kCostFunctionAUROC;
  }

  params.mode = do_regression ? kModeRegression : kModeClassification;

  auto model = ev::TextClassifierModelFactory::GetInstance()->CreateModel(
      model_type, params);

  ev::StringRef command(argv[optind++]);

  if (command == "learn") {
    if (optind + 2 > argc) {
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] learn DB-PATH CLASS-ID",
           argv[0]);
    }

    cantera::ColumnFileWriter output(
        ev::OpenFile(argv[optind++], O_WRONLY | O_APPEND | O_CREAT, 0666));

    const auto class_id = ev::StringToFloat(argv[optind++]);

    const bool use_stdin = (optind == argc);
    bool done = false;

    std::vector<std::pair<uint32_t, cantera::optional_string_view>> row;

    while (!done) {
      kj::Array<const char> buffer;

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
      row.emplace_back(0,
                       cantera::string_view{reinterpret_cast<const char*>(&hdr),
                                            sizeof(Header)});
      row.emplace_back(1, cantera::string_view{
                              reinterpret_cast<const char*>(doc_hashes.data()),
                              sizeof(doc_hashes[0]) * doc_hashes.size()});
      output.PutRow(row);
    }
  } else if (command == "batch-learn") {
    static const auto kFlushInterval = 128_z;
    size_t buffer_fill = 0;

    if (optind + 1 != argc)
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] batch-learn DB-PATH", argv[0]);

    cantera::ColumnFileWriter output(
        ev::OpenFile(argv[optind++], O_WRONLY | O_APPEND | O_CREAT, 0666));

    ev::ThreadPool thread_pool;

    float class_id;
    size_t length;

    // Used to make sure documents are output in the same order they were
    // provided.
    auto previous_future = thread_pool.Launch([] { return 0; });

    while (2 == scanf("%g %zu", &class_id, &length)) {
      KJ_REQUIRE(length > 0, length);

      auto buffer = kj::heapArray<char>(length);

      KJ_REQUIRE('\n' == getchar());

      auto ret = fread(buffer.begin(), 1, length, stdin);
      KJ_REQUIRE(ret == length, "fread failed");

      previous_future = thread_pool.Launch([
        buffer = std::move(buffer),
        class_id,
        &output,
        &buffer_fill,
        previous_future = std::move(previous_future)
      ]() mutable {
        std::vector<uint64_t> doc_hashes;
        Tokenize(buffer.begin(), buffer.end(), doc_hashes);

        buffer = nullptr;

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

          std::vector<std::pair<uint32_t, cantera::optional_string_view>> row;

          row.emplace_back(
              0, cantera::string_view{reinterpret_cast<const char*>(&hdr),
                                      sizeof(Header)});
          row.emplace_back(1,
                           cantera::string_view{
                               reinterpret_cast<const char*>(doc_hashes.data()),
                               sizeof(doc_hashes[0]) * doc_hashes.size()});
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

    model->Load(ev::OpenFile(argv[optind++], O_RDONLY));
    model->Print();
  } else if (command == "classify") {
    if (optind + 1 > argc) {
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] classify MODEL-PATH [PATH]...",
           argv[0]);
    }

    model->Load(ev::OpenFile(argv[optind++], O_RDONLY));

    const bool use_stdin = (optind == argc);
    bool done = false;

    std::vector<uint64_t> doc_hashes;

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

      doc_hashes.clear();
      Tokenize(buffer.begin(), buffer.end(), doc_hashes);
      std::sort(doc_hashes.begin(), doc_hashes.end());

      printf("%s\t%.9g\n", name, model->Classify(doc_hashes));
    }
  } else if (command == "batch-classify") {
    if (optind + 1 != argc)
      errx(EX_USAGE, "Usage: %s [OPTION]... [--] batch-classify MODEL-PATH",
           argv[0]);

    model->Load(ev::OpenFile(argv[optind++], O_RDONLY));

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
        model = model.get()
      ]() mutable {
        std::vector<uint64_t> doc_hashes;
        Tokenize(buffer.data(), buffer.data() + buffer.size(), doc_hashes);
        std::sort(doc_hashes.begin(), doc_hashes.end());

        const auto v = model->Classify(doc_hashes);

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

    model->Train(
        cantera::ColumnFileReader(ev::OpenFile(argv[optind++], O_RDONLY)));

    model->Save(
        ev::OpenFile(argv[optind++], O_WRONLY | O_CREAT | O_TRUNC, 0666));
  } else {
    KJ_FAIL_REQUIRE("Unknown command", command.str());
  }
} catch (kj::Exception e) {
  KJ_LOG(FATAL, e);
  return EXIT_FAILURE;
}
