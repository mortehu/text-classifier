#include <cctype>

#include "base/hash.h"
#include "base/string.h"
#include "base/stringref.h"
#include "tools/text-classifier/23andme.h"

namespace ev {

void Tokenize23AndMe(const char* begin, const char* end,
                     std::vector<uint64_t>& result) {
  while (begin != end) {
    const auto line_begin = begin;
    auto line_end = begin;

    while (line_end != end && *line_end != '\n') ++line_end;

    begin = line_end;
    if (begin != end) ++begin;

    if (*line_begin == '#') continue;

    auto line = ev::StringRef(line_begin, line_end);

    while (!line.empty() && std::isspace(line.back())) line.pop_back();

    auto row = ev::Explode(line, "\t");

    for (auto i = row.begin(); i != row.end();) {
      if (i->size() == 0)
        i = row.erase(i);
      else
        ++i;
    }

    if (row.size() > 5 || row.size() < 4) continue;

    auto feature =
        ev::Hash(row[1]) * 0x7fffffff + ev::Hash(row[2]) * 0x1fffffffffffffff;

    if (row.size() == 5) {
      if (row[3].size() != 1 || row[4].size() != 1) continue;

      if (row[3] == "-" && row[4] == "-") continue;

      char allelle[3];
      allelle[0] = row[3][0];
      allelle[1] = row[4][0];
      allelle[2] = 0;

      feature += ev::Hash(allelle);
    } else {
      if (row[3] == "--") continue;

      feature += ev::Hash(row[3]);
    }

    result.emplace_back(feature);
  }
}

}  // namespace ev
