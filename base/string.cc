#include "string.h"

#include <algorithm>
#include <cstdlib>

#include <err.h>
#include <stdarg.h>
#include <sysexits.h>

#include <kj/debug.h>

namespace ev {

std::string StringPrintf(const char* format, ...) {
  va_list args;
  char* buf;

  va_start(args, format);

  KJ_SYSCALL(vasprintf(&buf, format, args));

  std::string result(buf);
  free(buf);

  return result;
}

std::vector<ev::StringRef> Explode(const ev::StringRef& string,
                                   const ev::StringRef& delimiter,
                                   size_t limit) {
  std::vector<ev::StringRef> result;

  for (auto i = string.begin(); i != string.end();) {
    if (limit && result.size() + 1 == limit) {
      result.emplace_back(i, string.end());
      break;
    }

    auto d = std::search(i, string.end(), delimiter.begin(), delimiter.end());

    result.emplace_back(i, d);

    i = d;

    if (i != string.end()) i += delimiter.end() - delimiter.begin();
  }

  return result;
}

std::string Trim(std::string str) {
  auto begin = str.begin();
  while (begin != str.end() && std::isspace(*begin))
    ++begin;
  str.erase(str.begin(), begin);

  while (!str.empty() && std::isspace(str.back()))
    str.pop_back();

  return str;
}

std::string ToLower(std::string str) {
  for (auto& ch : str) ch = std::tolower(ch);
  return str;
}

int64_t StringToInt64(const char* string) {
  char* endptr = nullptr;
  long long value = strtoll(string, &endptr, 0);
  KJ_REQUIRE(*endptr == 0, "unexpected character in numeric string", string);
  return value;
}

uint64_t StringToUInt64(const char* string) {
  char* endptr = nullptr;
  unsigned long long value = strtoull(string, &endptr, 0);
  KJ_REQUIRE(*endptr == 0, "unexpected character in numeric string", string);
  return value;
}

double StringToDouble(const char* string) {
  char* endptr = nullptr;
  auto value = strtod(string, &endptr);
  KJ_REQUIRE(*endptr == 0, "unexpected character in numeric string", string);
  return value;
}

float StringToFloat(const char* string) {
  char* endptr = nullptr;
  auto value = strtof(string, &endptr);
  KJ_REQUIRE(*endptr == 0, "unexpected character in numeric string", string);
  return value;
}

std::string DoubleToString(const double v) {
  if (!v) return "0";

  if ((v >= 1e-6 || v <= -1e-6) && v < 1e17 && v > -1e17) {
    for (int prec = 0; prec < 17; ++prec) {
      auto result = StringPrintf("%.*f", prec, v);
      auto test_v = strtod(result.c_str(), nullptr);
      if (test_v == v) return result;
    }
  }

  return StringPrintf("%.17g", v);
}

std::string FloatToString(const float v) {
  if (!v) return "0";

  if ((v >= 1e-6 || v <= -1e-6) && v < 1e9 && v > -1e9) {
    for (int prec = 0; prec < 9; ++prec) {
      auto result = StringPrintf("%.*f", prec, v);
      auto test_v = strtof(result.c_str(), nullptr);
      if (test_v == v) return result;
    }
  }

  return StringPrintf("%.9g", v);
}

}  // namespace ev
