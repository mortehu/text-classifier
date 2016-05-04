#include <algorithm>
#include <cassert>
#include <unordered_set>

#include "base/cat.h"
#include "base/file.h"
#include "base/macros.h"
#include "base/string.h"
#include "base/thread-pool.h"
#include "index/web/tagsoup.h"
#include "third_party/gtest/gtest.h"

namespace {

ev::concurrency::RegionPool region_pool(4, 2048);

// The 10 top-sized categories.
const char* kTopics[] = {"earn", "trade", "acq",   "interest", "money-fx",
                         "ship", "grain", "wheat", "crude",    "corn"};

}  // namespace

struct ReutersTest : public testing::Test {
 public:
  static void SetUpTestCase() {
    for (int i = 0; i <= 21; ++i) {
      auto data = ev::ReadFile(
          ev::StringPrintf("tools/text-classifier/reuters/reut2-%03d.sgm", i)
              .c_str());
      ev::Tagsoup tagsoup(ev::StringRef{data.begin(), data.size()},
                          region_pool.GetRegion());

      ProcessNode(tagsoup.Root());
    }
  }

 protected:
  struct Document {
    bool test = false;
    bool skip = false;
    std::unordered_set<std::string> topics;
    std::string data;
    std::string date;
  };

  static void ProcessNode(const ev::TagsoupNode* node,
                          Document* doc = nullptr) {
    do {
      if (node->name == ev::TAGSOUP_NODE_ROOT) {
        if (node->first_child) {
          ProcessNode(node->first_child);
        }
      } else if (node->name == ev::TAGSOUP_NODE_CONTENT ||
                 node->name == ev::TAGSOUP_NODE_COMMENT) {
        if (doc) {
          if (!in_text_.empty() && in_text_.back()) {
            for (const auto ch : node->content) {
              switch (ch) {
                case '&':
                  doc->data += "&amp;";
                  break;
                case '<':
                  doc->data += "&lt;";
                  break;
                case '>':
                  doc->data += "&gt;";
                  break;
                default:
                  doc->data.push_back(ch);
              }
            }
          } else if (node_stack_.size() >= 2 && node_stack_.back() == "D") {
            if (node_stack_[node_stack_.size() - 2] == "TOPICS") {
              doc->topics.emplace(node->content.str());
            }
          } else if (node_stack_.back() == "DATE") {
            doc->date = node->content.str();
          }
        }
      } else {
        if (!in_text_.empty() && in_text_.back()) {
          doc->data += ev::cat("<", node->name, ">");
          ProcessNode(node->first_child, doc);
          doc->data += ev::cat("</", node->name, ">");
        } else if (node->name == "REUTERS") {
          Document new_doc;

          for (auto attr = node->first_attribute; attr;
               attr = attr->next_sibling) {
            if (attr->name == "LEWISSPLIT") {
              if (attr->content == "TEST")
                new_doc.test = true;
              else if (attr->content == "NOT-USED")
                new_doc.skip = true;
            }
          }

          if (node->first_child) {
            node_stack_.push_back(node->name);
            ProcessNode(node->first_child, &new_doc);
            node_stack_.pop_back();
          }

          documents_.emplace_back(std::move(new_doc));
        } else {
          if (node->first_child) {
            in_text_.emplace_back(node->name == "TEXT" ||
                                  (!in_text_.empty() && in_text_.back()));
            node_stack_.push_back(node->name);
            ProcessNode(node->first_child, doc);
            node_stack_.pop_back();
            in_text_.pop_back();
          }
        }
      }

      node = node->next_sibling;
    } while (node);
  }

  static std::vector<ev::StringRef> node_stack_;
  static std::vector<bool> in_text_;

  static std::vector<Document> documents_;
};

std::vector<ev::StringRef> ReutersTest::node_stack_;
std::vector<bool> ReutersTest::in_text_;
std::vector<ReutersTest::Document> ReutersTest::documents_;

TEST_F(ReutersTest, LearnCategories) {
  const auto working_dir = ev::TemporaryDirectory();

  ev::DirectoryTreeRemover working_dir_remover(working_dir);

  // Load data.

  for (const auto& topic : kTopics) {
    const auto topic_data_path = ev::cat(working_dir, "/", topic, ".data");

    ev::UniqueFILE learn(
        popen(ev::cat("tools/text-classifier/text-classifier "
                      "batch-learn --no-unique --strategy=html ",
                      topic_data_path)
                  .c_str(),
              "w"),
        &fclose);

    size_t training_count = 0;

    for (const auto& doc : documents_) {
      if (doc.skip || doc.test || doc.topics.empty()) continue;

      fprintf(learn.get(), "%.17g %zu\n", doc.topics.count(topic) ? 1.0 : 0.0,
              doc.data.size());
      ASSERT_EQ(doc.data.size(),
                fwrite(doc.data.data(), 1, doc.data.size(), learn.get()));

      ++training_count;
    }
  }

  // Train models in parallel.

  ev::ThreadPool thread_pool;

  for (const auto& topic : kTopics) {
    thread_pool.Launch([&] {
      const auto topic_data_path = ev::cat(working_dir, "/", topic, ".data");
      const auto topic_model_path = ev::cat(working_dir, "/", topic, ".model");
      ASSERT_EQ(0,
                system(ev::cat("tools/text-classifier/text-classifier analyze "
                               "--no-debug --weight=bns --threshold=3 --C=4 ",
                               topic_data_path, " ", topic_model_path)
                           .c_str()));
    });
  }

  thread_pool.Wait();

  // Run classification.

  auto sum_auc_roc = 0.0;

  for (const auto& topic : kTopics) {
    const auto topic_model_path = ev::cat(working_dir, "/", topic, ".model");
    const auto topic_output_path = ev::cat(working_dir, "/", topic, ".output");
    ev::UniqueFILE classify(
        popen(ev::cat("tools/text-classifier/text-classifier batch-classify ",
                      topic_model_path, " > ", topic_output_path)
                  .c_str(),
              "w"),
        &fclose);

    for (size_t i = 0; i < documents_.size(); ++i) {
      const auto& doc = documents_[i];

      if (doc.skip || !doc.test || doc.topics.empty()) continue;

      fprintf(classify.get(), "%zu %zu\n", i, doc.data.size());
      ASSERT_EQ(doc.data.size(),
                fwrite(doc.data.data(), 1, doc.data.size(), classify.get()));
    }

    // Wait for classification to finish.
    classify.reset();

    std::vector<std::pair<float, size_t> > results;
    size_t positive_count = 0;

    auto output = ev::OpenFileStream(topic_output_path.c_str(), "r");

    size_t idx;
    float score;
    while (2 == fscanf(output.get(), "%zu %f", &idx, &score)) {
      const auto target = documents_[idx].topics.count(topic);
      results.emplace_back(score, target);
      if (target) ++positive_count;
    }

    const auto negative_count = results.size() - positive_count;

    std::sort(results.begin(), results.end(),
              [](const auto& lhs, const auto& rhs) {
                return (lhs.first != rhs.first) ? (lhs.first < rhs.first)
                                                : (lhs.second > rhs.second);
              });

    auto auc_roc = 0.0;
    size_t seen_negative = 0;

    for (const auto& r : results) {
      if (r.second == 0) {
        ++seen_negative;
      } else {
        KJ_ASSERT(r.second == 1);
        auc_roc += static_cast<double>(seen_negative) /
                   (negative_count * positive_count);
      }
    }

    EXPECT_LT(0.98, auc_roc);

    fprintf(stderr, "%s: %.4f\n", topic, auc_roc);

    sum_auc_roc += auc_roc;
  }

  fprintf(stderr, "Mean: %.4f\n", sum_auc_roc / ARRAY_SIZE(kTopics));
}
