#include "base/string.h"
#include "base/thread-pool.h"
#include "index/web/tagsoup.h"
#include "tools/text-classifier/html-tokenizer.h"

namespace {

ev::concurrency::RegionPool region_pool(256, 2048);

static const char* kStopWords[] = {
    "a",          "about",     "above",     "after",   "again",    "against",
    "all",        "am",        "an",        "and",     "any",      "are",
    "aren't",     "as",        "at",        "be",      "because",  "been",
    "before",     "being",     "below",     "between", "both",     "but",
    "by",         "can't",     "cannot",    "could",   "couldn't", "did",
    "didn't",     "do",        "does",      "doesn't", "doing",    "don't",
    "down",       "during",    "each",      "few",     "for",      "from",
    "further",    "had",       "hadn't",    "has",     "hasn't",   "have",
    "haven't",    "having",    "he",        "he'd",    "he'll",    "he's",
    "her",        "here",      "here's",    "hers",    "herself",  "him",
    "himself",    "his",       "how",       "how's",   "i",        "i'd",
    "i'll",       "i'm",       "i've",      "if",      "in",       "into",
    "is",         "isn't",     "it",        "it's",    "its",      "itself",
    "let's",      "me",        "more",      "most",    "mustn't",  "my",
    "myself",     "no",        "nor",       "not",     "of",       "off",
    "on",         "once",      "only",      "or",      "other",    "ought",
    "our",        "ours",      "ourselves", "out",     "over",     "own",
    "same",       "shan't",    "she",       "she'd",   "she'll",   "she's",
    "should",     "shouldn't", "so",        "some",    "such",     "than",
    "that",       "that's",    "the",       "their",   "theirs",   "them",
    "themselves", "then",      "there",     "there's", "these",    "they",
    "they'd",     "they'll",   "they're",   "they've", "this",     "those",
    "through",    "to",        "too",       "under",   "until",    "up",
    "very",       "was",       "wasn't",    "we",      "we'd",     "we'll",
    "we're",      "we've",     "were",      "weren't", "what",     "what's",
    "when",       "when's",    "where",     "where's", "which",    "while",
    "who",        "who's",     "whom",      "why",     "why's",    "with",
    "won't",      "would",     "wouldn't",  "you",     "you'd",    "you'll",
    "you're",     "you've",    "your",      "yours",   "yourself", "yourselves",
};

}  // namespace

namespace ev {

HTMLTokenizer::HTMLTokenizer() {
  for (const auto stop_word : kStopWords) stop_words_.emplace(stop_word);
}

void HTMLTokenizer::Tokenize(const ev::TagsoupNode* node,
                             std::vector<uint64_t>& result) {
  static const std::array<uint64_t, 2> kWindowMultipliers{{3, 5}};

  do {
    if (node->name == ev::TAGSOUP_NODE_ROOT) {
      if (node->first_child) {
        Tokenize(node->first_child, result);
      }
    } else if (node->name == ev::TAGSOUP_NODE_CONTENT ||
               node->name == ev::TAGSOUP_NODE_COMMENT) {
      auto token_begin = node->content.begin();
      const auto end = node->content.end();

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

        if (stop_words_.count(token)) continue;

        const auto token_hash = ev::Hash(token);

        result.emplace_back(token_hash);

        for (size_t i = 0; i < window_.size(); ++i)
          result.emplace_back(token_hash + kWindowMultipliers[i] * window_[i]);

        window_.emplace_front(token_hash);
        if (window_.size() > kWindowMultipliers.size()) window_.pop_back();

        if (!node_stack_.empty())
          result.emplace_back(node_stack_.back() + token_hash);
      }
    } else {
      uint64_t name_hash = ev::Hash(node->name) * 103;
      TagsoupAttribute* processed_attribute = nullptr;

      // Some tags' name attributes are combined into the tag name.
      if (node->name == "meta") {
        for (auto attr = node->first_attribute; attr;
             attr = attr->next_sibling) {
          if (attr->name == "name" || attr->name == "property" ||
              attr->name == "http-equiv") {
            name_hash += ev::Hash(ev::ToLower(attr->content.str())) * 29;
            processed_attribute = attr;
            break;
          }
        }
      } else if (node->name == "link") {
        for (auto attr = node->first_attribute; attr;
             attr = attr->next_sibling) {
          if (attr->name == "rel") {
            name_hash += ev::Hash(ev::ToLower(attr->content.str())) * 29;
            processed_attribute = attr;
            break;
          }
        }
      }

      result.emplace_back(name_hash);

      if (!node_stack_.empty())
        result.emplace_back(name_hash * 31 + node_stack_.back());

      for (auto attr = node->first_attribute; attr; attr = attr->next_sibling) {
        if (attr == processed_attribute) continue;

        const auto attr_name_hash = ev::Hash(attr->name) * 41 + name_hash;
        const auto attr_content_hash = ev::Hash(attr->content) * 71;

        result.emplace_back(attr_name_hash);
        result.emplace_back(attr_name_hash + attr_content_hash);

        if (ev::HasPrefix(attr->content, "http://") ||
            ev::HasPrefix(attr->content, "https://")) {
          const auto& uri = attr->content;

          auto component_begin = uri.begin();
          while (component_begin != uri.end()) {
            if (*component_begin == '/') {
              ++component_begin;
              continue;
            }

            auto component_end = uri.find('/', component_begin - uri.begin());

            const auto component_hash =
                ev::Hash(ev::StringRef(component_begin, component_end)) * 131;
            result.emplace_back(component_hash);
            result.emplace_back(attr_name_hash + component_hash);

            component_begin = component_end;
          }
        }
      }

      if (node->first_child) {
        node_stack_.emplace_back(name_hash);
        Tokenize(node->first_child, result);
        node_stack_.pop_back();
      }
    }

    node = node->next_sibling;
  } while (node);
}

void HTMLTokenizer::Tokenize(const char* begin, const char* end,
                             std::vector<uint64_t>& result) {
  ev::Tagsoup tagsoup(ev::StringRef(begin, end), region_pool.GetRegion());

  Tokenize(tagsoup.Root(), result);
}

}  // namespace ev
