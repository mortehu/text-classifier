#ifndef TOOLS_TEXT_CLASSIFIER_23ANDME_H_
#define TOOLS_TEXT_CLASSIFIER_23ANDME_H_ 1

#include <cstdint>
#include <vector>

namespace ev {

void Tokenize23AndMe(const char* begin, const char* end,
                     std::vector<uint64_t>& result);

}  // namespace ev

#endif  // !TOOLS_TEXT_CLASSIFIER_23ANDME_H_
