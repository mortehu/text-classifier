#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "base/columnfile.h"

#include <fcntl.h>
#include <unistd.h>

#include <kj/array.h>
#include <kj/debug.h>

#if HAVE_LIBSNAPPY
#include <snappy.h>
#endif

#include "base/columnfile-internal.h"
#include "base/file.h"
#include "base/macros.h"

namespace ev {

namespace {

using namespace columnfile_internal;

uint32_t GetUInt(StringRef& input) {
  auto begin = reinterpret_cast<const uint8_t*>(input.begin());
  auto i = begin;
  uint32_t b = *i++;
  uint32_t result = b & 127;
  if (b < 0x80) goto done;
  b = *i++;
  result |= (b & 127) << 6;
  if (b < 0x80) goto done;
  b = *i++;
  result |= (b & 127) << 13;
  if (b < 0x80) goto done;
  b = *i++;
  result |= (b & 127) << 20;
  if (b < 0x80) goto done;
  b = *i++;
  result |= b << 27;
done:
  input.Consume(i - begin);
  return result;
}

int32_t GetInt(StringRef& input) {
  const auto u = GetUInt(input);
  return (u >> 1) ^ -((int32_t)(u & 1));
}

class ColumnFileFdInput : public ColumnFileInput {
 public:
  ColumnFileFdInput(kj::AutoCloseFd fd) : fd_(std::move(fd)) {
    (void)posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);

    char magic[sizeof(kMagic)];
    Read(fd_, magic, sizeof(kMagic), sizeof(kMagic));
    KJ_REQUIRE(!memcmp(magic, kMagic, sizeof(kMagic)));
  }

  ~ColumnFileFdInput() override {}

  std::vector<std::pair<uint32_t, kj::Array<const char>>> Fill(
      const std::unordered_set<uint32_t>& field_filter,
      ColumnFileCompression& compression) override;

  bool End() const override { return end_; }

  void SeekToStart() override {
    KJ_REQUIRE(fd_ != nullptr);

    KJ_SYSCALL(lseek(fd_, sizeof(kMagic), SEEK_SET));

    data_.clear();
    buffer_.clear();
    end_ = false;
  }

 private:
  bool end_ = false;

  std::string buffer_;

  ev::StringRef data_;

  kj::AutoCloseFd fd_;
};

class ColumnFileStringInput : public ColumnFileInput {
 public:
  ColumnFileStringInput(ev::StringRef data) : input_data_(data) {
    KJ_REQUIRE(input_data_.size() >= sizeof(kMagic));
    KJ_REQUIRE(!memcmp(input_data_.begin(), kMagic, sizeof(kMagic)));
    input_data_.Consume(sizeof(kMagic));

    data_ = input_data_;
  }

  ~ColumnFileStringInput() override {}

  std::vector<std::pair<uint32_t, kj::Array<const char>>> Fill(
      const std::unordered_set<uint32_t>& field_filter,
      ColumnFileCompression& compression) override;

  bool End() const override { return data_.empty(); }

  void SeekToStart() override { data_ = input_data_; }

 private:
  ev::StringRef input_data_;
  ev::StringRef data_;
};

std::vector<std::pair<uint32_t, kj::Array<const char>>> ColumnFileFdInput::Fill(
    const std::unordered_set<uint32_t>& field_filter,
    ColumnFileCompression& compression) {
  std::vector<std::pair<uint32_t, kj::Array<const char>>> result;

  KJ_REQUIRE(data_.empty());

  uint8_t size_buffer[4];
  auto ret = Read(fd_, size_buffer, 0, 4);
  if (ret < 4) {
    end_ = true;
    KJ_REQUIRE(ret == 0);
    return result;
  }

  auto size = (size_buffer[0] << 24) | (size_buffer[1] << 16) |
              (size_buffer[2] << 8) | size_buffer[3];
  buffer_.resize(size);
  Read(fd_, &buffer_[0], size, size);

  data_ = buffer_;

  compression = static_cast<ColumnFileCompression>(GetUInt(data_));

  std::vector<std::pair<uint32_t, uint32_t>> field_sizes;

  auto field_count = GetUInt(data_);

  field_sizes.reserve(field_count);
  result.reserve(field_filter.empty() ? field_count : field_filter.size());

  for (auto i = field_count; i-- > 0;) {
    auto field_idx = GetUInt(data_);
    auto field_size = GetUInt(data_);
    field_sizes.emplace_back(field_idx, field_size);
  }

  // Number of bytes to seek before next read.  The purpose of having this
  // variable is to avoid calling lseek several times back-to-back on the
  // same file descriptor.
  size_t seek_amount = 0;

  for (const auto& f : field_sizes) {
    // If the field is ignored, skip its data.
    if (!field_filter.empty() && !field_filter.count(f.first)) {
      seek_amount += f.second;
      continue;
    }

    if (seek_amount > 0) {
      KJ_SYSCALL(lseek(fd_, seek_amount, SEEK_CUR));
      seek_amount = 0;
    }

    auto buffer = kj::heapArray<char>(f.second);
    Read(fd_, buffer.begin(), f.second, f.second);

    result.emplace_back(f.first, std::move(buffer));
  }

  if (seek_amount > 0) {
    KJ_SYSCALL(lseek(fd_, seek_amount, SEEK_CUR));
  }

  return std::move(result);
}

std::vector<std::pair<uint32_t, kj::Array<const char>>>
ColumnFileStringInput::Fill(const std::unordered_set<uint32_t>& field_filter,
                            ColumnFileCompression& compression) {
  KJ_REQUIRE(!data_.empty());

  data_.Consume(4);  // Skip header size we don't need.

  compression = static_cast<ColumnFileCompression>(GetUInt(data_));

  std::vector<std::pair<uint32_t, uint32_t>> field_sizes;

  auto field_count = GetUInt(data_);
  for (auto i = field_count; i-- > 0;) {
    auto field_idx = GetUInt(data_);
    auto field_size = GetUInt(data_);
    field_sizes.emplace_back(field_idx, field_size);
  }

  std::vector<std::pair<uint32_t, kj::Array<const char>>> result;

  for (const auto& f : field_sizes) {
    // If the field is ignored, skip its data.
    if (!field_filter.empty() && !field_filter.count(f.first)) {
      data_.Consume(f.second);
      continue;
    }

    // TODO(mortehu): See if we can use a non-owning array instead, e.g.
    //     kj::Array<const char>(data_.begin(), f.second,
    //     kj::NullArrayDisposer());
    //  This way, we wouldn't have to copy all the data.

    KJ_REQUIRE(data_.size() >= f.second, data_.size());
    auto buffer = kj::heapArray<char>(data_.begin(), f.second);
    data_.Consume(f.second);

    result.emplace_back(f.first, std::move(buffer));
  }

  return std::move(result);
}

}  // namespace

ColumnFileReader::ColumnFileReader(std::unique_ptr<ColumnFileInput> input)
    : input_(std::move(input)) {}

ColumnFileReader::ColumnFileReader(kj::AutoCloseFd fd)
    : input_(std::make_unique<ColumnFileFdInput>(std::move(fd))) {}

ColumnFileReader::ColumnFileReader(StringRef input)
    : input_(std::make_unique<ColumnFileStringInput>(input)) {}

void ColumnFileReader::SetColumnFilter(
    std::initializer_list<uint32_t> columns) {
  column_filter_.clear();
  for (auto column : columns) column_filter_.emplace(column);
}

bool ColumnFileReader::End() {
  for (auto i = fields_.begin(); i != fields_.end(); i = fields_.erase(i)) {
    if (!i->second.End()) return false;
  }

  if (input_->End()) return true;

  Fill();

  return fields_.empty();
}

const StringRef* ColumnFileReader::Peek(uint32_t field) {
  for (auto i = fields_.begin(); i != fields_.end();) {
    if (i->second.End())
      i = fields_.erase(i);
    else
      ++i;
  }

  if (fields_.empty()) Fill();

  auto i = fields_.find(field);
  KJ_REQUIRE(i != fields_.end(), "Missing field", field);

  return i->second.Peek();
}

const StringRef* ColumnFileReader::Get(uint32_t field) {
  for (auto i = fields_.begin(); i != fields_.end();) {
    if (i->second.End())
      i = fields_.erase(i);
    else
      ++i;
  }

  if (fields_.empty()) Fill();

  auto i = fields_.find(field);
  KJ_REQUIRE(i != fields_.end(), "Missing field", field);

  return i->second.Get();
}

const std::vector<std::pair<uint32_t, StringRef>>& ColumnFileReader::GetRow() {
  row_buffer_.clear();

  // TODO(mortehu): This function needs optimization.

  for (auto i = fields_.begin(); i != fields_.end();) {
    if (i->second.End())
      i = fields_.erase(i);
    else
      ++i;
  }

  if (fields_.empty()) Fill();

  if (row_buffer_.capacity() < fields_.size())
    row_buffer_.reserve(fields_.size());

  for (auto i = fields_.begin(); i != fields_.end(); ++i) {
    auto data = i->second.Get();

    if (data) row_buffer_.emplace_back(i->first, *data);
  }

  return row_buffer_;
}

void ColumnFileReader::SeekToStart() {
  input_->SeekToStart();

  fields_.clear();
  row_buffer_.clear();
}

ColumnFileReader::FieldReader::FieldReader(kj::Array<const char> buffer,
                                           ColumnFileCompression compression)
    : buffer_(std::move(buffer)), data_(buffer_), compression_(compression) {}

void ColumnFileReader::FieldReader::Fill() {
  switch (compression_) {
    case kColumnFileCompressionNone:
      break;

#if HAVE_LIBSNAPPY
    case kColumnFileCompressionSnappy: {
      KJ_REQUIRE(snappy::IsValidCompressedBuffer(data_.data(), data_.size()));
      size_t decompressed_size = 0;
      KJ_REQUIRE(snappy::GetUncompressedLength(data_.data(), data_.size(),
                                               &decompressed_size));

      auto decompressed_data = kj::heapArray<char>(decompressed_size);
      KJ_REQUIRE(snappy::RawUncompress(data_.data(), data_.size(),
                                       decompressed_data.begin()));
    buffer_ = std::move(decompressed_data);

    data_ = buffer_;
    compression_ = kColumnFileCompressionNone;
    } break;
#endif

    default:
      KJ_FAIL_REQUIRE("Unsupported compression scheme", compression_);
  }

  if (!repeat_) {
    repeat_ = GetUInt(data_);
    KJ_REQUIRE(0 == GetUInt(data_));  // Reserved field

    auto b0 = static_cast<uint8_t>(data_[0]);

    if ((b0 & 0xc0) == 0xc0) {
      data_.Consume(1);
      if (b0 == kCodeNull) {
        value_is_null_ = true;
      } else {
        // The value we're about to read shares a prefix at least 2 bytes long
        // with the previous value.
        auto shared_prefix = (b0 & 0x3f) + 2;
        auto suffix_length = GetUInt(data_);

        // We just move the old prefix in front of the new suffix, corrupting
        // whatever data is there; we're not going to read it again anyway.
        memmove(const_cast<char*>(data_.begin()) - shared_prefix,
                value_.begin(), shared_prefix);

        value_ = StringRef(data_.begin() - shared_prefix,
                           shared_prefix + suffix_length);
        data_.Consume(suffix_length);
        value_is_null_ = false;
      }
    } else {
      auto value_size = GetUInt(data_);
      value_ = StringRef(data_.begin(), value_size);
      data_.Consume(value_size);
      value_is_null_ = false;
    }
  }
}

void ColumnFileReader::Fill() {
  fields_.clear();

  ColumnFileCompression compression;
  auto fields = input_->Fill(column_filter_, compression);

  if (fields.empty()) return;

  for (auto& field : fields) {
    fields_.emplace(field.first,
                    FieldReader(std::move(field.second), compression));
  }
}

}  // namespace ev
