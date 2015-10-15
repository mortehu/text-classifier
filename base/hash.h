#ifndef BASE_HASH_H_
#define BASE_HASH_H_ 1

#include <cstdint>

namespace ev {

class StringRef;

uint64_t Hash(const StringRef& key);

}  // namespace ev

#endif  // !BASE_HASH_H
