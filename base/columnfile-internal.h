#ifndef BASE_COLUMNFILE_INTERNAL_H_
#define BASE_COLUMNFILE_INTERNAL_H_ 1

namespace ev {
namespace columnfile_internal {

// The magic code string is designed to cause a parse error if someone attempts
// to parse the file as a CSV.
static const char kMagic[4] = {'\n', '\t', '\"', 0};

enum Codes : uint8_t {
  kCodeNull = 0xff,
};

}  // namespace columnfile_internal
}  // namespace ev

#endif // !BASE_COLUMNFILE_INTERNAL_H_
