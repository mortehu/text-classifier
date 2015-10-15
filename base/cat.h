#ifndef BASE_CAT_H_
#define BASE_CAT_H_

#include <cstdio>
#include <cstring>
#include <limits>

#include <kj/string.h>

#include "base/stringref.h"

namespace ev {

// Instead of explicitly creating lots of specializations of `cat_to()', we use
// specializations of this struct to generate them.
template <typename Type>
struct CatPrintfFormat {
  // Denotes whether we have defined a printf format string for `Type'.
  static constexpr bool is_specialized = false;

  // Format string to pass to `printf()' to print values of type `Type'.
  static constexpr const char* format_string = nullptr;

  // Size in bytes of buffer needed to format `Type'.
  static constexpr size_t buffer_size = 0;
};

#define FMT(type_, format_string_, buffer_size_)                   \
  template <>                                                      \
  struct CatPrintfFormat<type_> {                                  \
    static constexpr bool is_specialized = true;                   \
    static constexpr const char* format_string = (format_string_); \
    static constexpr size_t buffer_size = (buffer_size_);          \
  }

FMT(char, "%c", 2);
FMT(short, "%d", std::numeric_limits<short>::digits10 + 3);
FMT(int, "%d", std::numeric_limits<int>::digits10 + 3);
FMT(long, "%ld", std::numeric_limits<long>::digits10 + 3);
FMT(long long, "%lld", std::numeric_limits<long long>::digits10 + 3);

FMT(unsigned char, "%u", std::numeric_limits<unsigned char>::digits10 + 2);
FMT(unsigned short, "%u", std::numeric_limits<unsigned short>::digits10 + 2);
FMT(unsigned int, "%u", std::numeric_limits<unsigned int>::digits10 + 2);
FMT(unsigned long, "%lu", std::numeric_limits<unsigned long>::digits10 + 2);
FMT(unsigned long long, "%llu",
    std::numeric_limits<unsigned long long>::digits10 + 2);

FMT(float, "%.9g", 15);
FMT(double, "%.17g", 24);

#undef FMT

template <typename Container>
void format_to(Container* output, const std::string& arg) {
  output->insert(output->end(), arg.begin(), arg.end());
}

template <typename Container>
void format_to(Container* output, char* arg) {
  output->insert(output->end(), arg, arg + std::strlen(arg));
}

template <typename Container>
void format_to(Container* output, const char* arg) {
  output->insert(output->end(), arg, arg + std::strlen(arg));
}

template <typename Container>
void format_to(Container* output, StringRef arg) {
  output->insert(output->end(), arg.begin(), arg.end());
}

template <typename Container>
void format_to(Container* output, const kj::StringPtr& arg) {
  output->insert(output->end(), arg.begin(), arg.end());
}

template <typename Container, typename T>
void format_to(Container* output, T arg) {
  static_assert(CatPrintfFormat<T>::is_specialized,
                "don't know how to print this type");
  char buffer[CatPrintfFormat<T>::buffer_size];
  size_t length = std::snprintf(buffer, sizeof(buffer),
                                CatPrintfFormat<T>::format_string, arg);
  output->insert(output->end(), buffer, buffer + length);
}

template <typename Container>
void cat_to(Container* output) {}

template <typename Container, typename T, typename... Args>
void cat_to(Container* output, T arg1, const Args&... args) {
  format_to(output, arg1);
  cat_to(output, args...);
}

template <typename... Args>
std::string cat(const Args&... args) {
  std::string result;
  cat_to(&result, args...);
  return result;
}

}  // namespace ev

#endif  // !BASE_CAT_H_
