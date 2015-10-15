#ifndef BASE_STRING_H_
#define BASE_STRING_H_

#include <string>

#include <kj/debug.h>

#include "base/stringref.h"

namespace ev {

std::string StringPrintf(const char* format, ...);

inline bool HasPrefix(const ev::StringRef& haystack,
                      const ev::StringRef& needle) {
  if (haystack.size() < needle.size()) return false;
  return 0 == memcmp(haystack.begin(), needle.begin(), needle.size());
}

inline bool HasSuffix(const ev::StringRef& haystack,
                      const ev::StringRef& needle) {
  if (haystack.size() < needle.size()) return false;
  return 0 == memcmp(haystack.begin() + haystack.size() - needle.size(),
                     needle.begin(), needle.size());
}

inline void AppendTo(std::string& buffer, const char* data, size_t length) {
  buffer.insert(buffer.end(), data, data + length);
}

std::vector<ev::StringRef> Explode(const ev::StringRef& string,
                                   const ev::StringRef& delimiter,
                                   size_t limit = 0);

// Removes leading and trailing white-space from `str`, as determined by
// std::isspace().
std::string Trim(std::string str);

// Converts all character in `str` to lower-case, using std::tolower().
std::string ToLower(std::string str);

template <typename T>
void BinaryToHex(const uint8_t* input, size_t size, T* output) {
  static const char kHexDigits[] = "0123456789abcdef";

  for (size_t i = 0; i < size; ++i) {
    output->push_back(kHexDigits[input[i] >> 4]);
    output->push_back(kHexDigits[input[i] & 15]);
  }
}

template <typename InputIterator, typename OutputIterator>
void HexToBinary(InputIterator begin, InputIterator end,
                 OutputIterator output) {
  // This LUT takes advantage of the fact that the lower 5 bits in the ASCII
  // representation of all hexadecimal digits are unique.
  static const uint8_t kHexHelper[26] = {0, 10, 11, 12, 13, 14, 15, 0, 0,
                                         0, 0,  0,  0,  0,  0,  0,  0, 1,
                                         2, 3,  4,  5,  6,  7,  8,  9};

  while (begin != end) {
    auto c0 = *begin++;
    if (begin == end)
      KJ_FAIL_REQUIRE("hexadecimal number has odd number of digits");
    auto c1 = *begin++;

    if (!std::isxdigit(c0) || !std::isxdigit(c1))
      KJ_FAIL_REQUIRE("input is not hexadecimal");

    *output++ = (kHexHelper[c0 & 0x1f] << 4) | (kHexHelper[c1 & 0x1f]);
  }
}

int64_t StringToInt64(const char* string);
uint64_t StringToUInt64(const char* string);

double StringToDouble(const char* string);
float StringToFloat(const char* string);

// Formats a floating point value as a string, preserving enough digits to
// reconstruct it exactly.
std::string DoubleToString(const double v);
std::string FloatToString(const float v);

}  // namespace ev

#endif  // !BASE_STRING_H_
