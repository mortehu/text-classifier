#include "base/hash.h"

#include "base/stringref.h"

namespace ev {

// This function is based on std::hash<std::string> from version 4.9 of the GNU
// C++ library, and produces the exact same output.  It is replicated here
// because std::hash<> is not guaranteed to be the same function even across
// different invocations of the same program.
uint64_t Hash(const StringRef& key) {
  static const uint64_t mul = (0xc6a4a793UL << 32UL) + 0x5bd1e995UL;

  static_assert(sizeof(uint64_t) == 8, "uint64_t must be 8 bytes");

  auto shift_mix = [](uint64_t v) -> uint64_t { return v ^ (v >> 47); };

  auto len = key.size();
  auto buf = key.data();

  const auto len_aligned = len & ~7;
  const auto end = buf + len_aligned;
  uint64_t hash = 0xc70f6907 ^ (len * mul);
  auto p = buf;

  for (; p != end; p += 8) {
    uint64_t tmp;
    memcpy(&tmp, p, sizeof(uint64_t));
    const auto data = shift_mix(tmp * mul) * mul;
    hash ^= data;
    hash *= mul;
  }

  len &= 7;

  if (len) {
    p += len;
    uint64_t data = 0;
    while (len--) data = (data << 8) + static_cast<unsigned char>(*--p);

    hash ^= data;
    hash *= mul;
  }

  hash = shift_mix(hash) * mul;
  hash = shift_mix(hash);

  return hash;
}

}  // namespace ev
