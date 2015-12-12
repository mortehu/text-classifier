#ifndef HTML_TOKENIZER_H_
#define HTML_TOKENIZER_H_ 1

#include <cstdint>
#include <deque>
#include <unordered_set>
#include <vector>

#include "base/stringref.h"

namespace ev {

struct TagsoupNode;

class HTMLTokenizer {
 public:
  HTMLTokenizer();

  void Tokenize(const ev::TagsoupNode* node, std::vector<uint64_t>& result);

  void Tokenize(const char* begin, const char* end,
                std::vector<uint64_t>& result);

 private:
  std::deque<uint64_t> window_;

  std::unordered_set<ev::StringRef> stop_words_;

  std::vector<uint64_t> node_stack_;
};

}  // namespace ev

#endif  // !HTML_TOKENIZER_H_
