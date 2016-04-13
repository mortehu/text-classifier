#ifndef BASE_STRINGREF_H_
#define BASE_STRINGREF_H_ 1

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <sys/uio.h>

#include <kj/array.h>
#include <kj/string.h>

#include "base/hash.h"
#include "base/region.h"

namespace ev {

// Holds a non-owning reference to an array of bytes.  Designed to allow
// functions to accept a multitude of string types, without unnecessary buffer
// allocations.
class StringRef {
 public:
  // Constucts an empty string.
  StringRef() = default;

  StringRef(const char* str, size_t size) : data_(str), size_(size) {}

  StringRef(const char* begin, const char* end)
      : data_(begin), size_(end - begin) {}

  StringRef(const char* str) : data_(str), size_(std::strlen(str)) {}

  StringRef(const std::string& str) : data_(str.data()), size_(str.size()) {}

  StringRef(const std::vector<char>& v) : data_(&v[0]), size_(v.size()) {}

  StringRef(const std::vector<unsigned char>& v)
      : data_(reinterpret_cast<const char*>(&v[0])), size_(v.size()) {}

  StringRef(const struct iovec& iov)
      : data_(reinterpret_cast<const char*>(iov.iov_base)),
        size_(iov.iov_len) {}

  StringRef(const kj::Array<const char>& buffer)
      : data_(buffer.begin()), size_(buffer.size()) {}

  StringRef(const kj::StringPtr& kjstr)
      : data_(kjstr.cStr()), size_(kjstr.size()) {}

  StringRef(std::nullptr_t) : data_(nullptr), size_(0) {}

  std::string str() const { return std::string(begin(), end()); }

  StringRef dup(ev::concurrency::RegionPool::Region& region) const {
    char* data = nullptr;
    if (size_) {
      data = reinterpret_cast<char*>(region.EphemeralAllocate(size_));
      memcpy(data, data_, size_);
    }
    return StringRef(data, size_);
  }

  explicit operator bool() const { return data_ != nullptr; }

  void clear() {
    data_ = nullptr;
    size_ = 0;
  }

  void pop_front() {
    ++data_;
    --size_;
  }

  void pop_back() { --size_; }

  // Similar to pop_front(), but consumes multiple elements.
  void Consume(size_t amount) {
    KJ_ASSERT(amount <= size_);
    data_ += amount;
    size_ -= amount;
  }

  // Similar to pop_back(), but consumes multiple elements.
  void ConsumeTail(size_t amount) { size_ -= amount; }

  char front() const { return data_[0]; }

  char back() const { return data_[size_ - 1]; }

  char operator[](size_t idx) const { return data_[idx]; }

  bool empty() const { return size_ == 0; }

  size_t size() const { return size_; }

  const char* data() const { return data_; }

  const char* begin() const { return data_; }

  const char* end() const { return data_ + size_; }

  // Returns a pointer to the first occurrence of the given string,
  // starting at `from`.  Returns `end()` if no match is found.
  const char* find(const ev::StringRef& needle, size_t from = 0) const {
    auto result =
        memmem(data_ + from, size_ - from, needle.data(), needle.size());

    return result ? reinterpret_cast<const char*>(result) : end();
  }

  // Returns a pointer to the first occurrence of the given character, starting
  // at `from`.  Returns `end()` if no match is found.
  const char* find(char ch, size_t from = 0) const {
    const char* result =
        reinterpret_cast<const char*>(memchr(data_ + from, ch, size_ - from));

    if (!result) return end();

    return result;
  }

  // Returns a pointer to the last occurrence of the given character in a
  // string, or `end()` if no match is found.
  const char* rfind(char ch) const {
    for (auto result = end(); result-- != begin(); ) {
      if (*result == ch) return result;
    }

    return end();
  }

  const char* find_first_of(const char* set, size_t from = 0) const {
    auto result = reinterpret_cast<const char*>(data_ + from);

    while (result != end()) {
      if (strchr(set, *result)) return result;
      ++result;
    }

    return end();
  }

  StringRef substr(size_t offset,
                   size_t n = std::numeric_limits<size_t>::max()) const {
    if (offset >= size_) return StringRef("", size_t());

    size_t tail_size = size_ - offset;
    if (n > tail_size) n = tail_size;

    return StringRef(data_ + offset, n);
  }

  int compare(const char* rhs) const {
    auto lhs = begin();

    while (lhs != end() && *rhs) {
      if (*lhs != *rhs) return *lhs - *rhs;
      ++lhs, ++rhs;
    }

    if (lhs == end()) return *rhs ? -1 : 0;

    return 1;
  }

  int compare(const StringRef& rhs) const {
    auto cmp = std::memcmp(data_, rhs.data_, std::min(size_, rhs.size_));
    if (cmp != 0) return cmp;
    return (size_ < rhs.size_) ? -1 : (size_ > rhs.size_) ? 1 : 0;
  }

  bool operator==(std::nullptr_t) const { return data_ == nullptr; }

  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }

  bool equals_lower(const StringRef& rhs) const {
    if (size_ != rhs.size_) return false;
    for (size_t i = 0; i < size_; ++i) {
      if (std::tolower(data_[i]) != std::tolower(rhs.data_[i])) return false;
    }
    return true;
  }

  bool operator==(const char* rhs) const {
    for (size_t i = 0; i < size_; ++i) {
      if (rhs[i] != data_[i]) return false;
    }
    return (rhs[size_] == 0);
  }

  bool operator==(const StringRef& rhs) const {
    if (size_ != rhs.size_) return false;
    return 0 == std::memcmp(data_, rhs.data_, size_);
  }

  bool operator!=(const StringRef& rhs) const {
    if (size_ != rhs.size_) return true;
    return 0 != std::memcmp(data_, rhs.data_, size_);
  }

  bool operator<(const StringRef& rhs) const {
    return std::lexicographical_compare(data_, data_ + size_, rhs.data_,
                                        rhs.data_ + rhs.size_);
  }

  bool operator>(const StringRef& rhs) const { return rhs < *this; }

  bool contains(const StringRef& rhs) const {
    return nullptr != memmem(data_, size_, rhs.data(), rhs.size());
  }

 private:
  const char* data_ = nullptr;
  size_t size_ = 0;
};

inline ev::StringRef operator*(kj::_::Stringifier, ev::StringRef str) {
  return str;
}

class StringRefOrNull {
 public:
  StringRefOrNull(std::nullptr_t) : null_(true) {}
  StringRefOrNull(ev::StringRef str) : string_(str) {}

  bool IsNull() const { return null_; }

  const ev::StringRef& StringRef() const {
    KJ_ASSERT(!null_);
    return string_;
  }

  bool operator==(const ev::StringRefOrNull& rhs) const {
    if (null_) return rhs.null_;
    if (rhs.null_) return false;
    return string_ == rhs.string_;
  }

  bool operator!=(const ev::StringRefOrNull& rhs) const {
    return !(*this == rhs);
  }

 private:
  bool null_ = false;
  ev::StringRef string_;
};

}  // namespace ev

namespace std {

template <>
struct hash<ev::StringRef> {
  typedef ev::StringRef argument_type;
  typedef std::size_t result_type;

  result_type operator()(argument_type const& str) const {
    return ev::Hash(str);
  }
};

}  // namespace std

#endif  // !BASE_STRINGREF_H_
