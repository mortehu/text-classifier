#include "base/columnfile.h"

#include <fcntl.h>
#include <unistd.h>

#include <kj/array.h>
#include <kj/common.h>
#include <kj/debug.h>

#if HAVE_LZ4
#include <lz4.h>
#endif

#if HAVE_LZMA
#include <lzma.h>
#endif

#if HAVE_LIBSNAPPY
#include <snappy.h>
#endif

#if HAVE_ZLIB
#include <zlib.h>
#endif

#include "base/columnfile-internal.h"
#include "base/file.h"
#include "base/macros.h"
#include "base/thread-pool.h"

namespace ev {

namespace {

using namespace columnfile_internal;

class ColumnFileFdInput : public ColumnFileInput {
 public:
  ColumnFileFdInput(kj::AutoCloseFd fd) : fd_(std::move(fd)) {
    (void)posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);

    char magic[sizeof(kMagic)];
    Read(fd_, magic, sizeof(kMagic), sizeof(kMagic));
    KJ_REQUIRE(!memcmp(magic, kMagic, sizeof(kMagic)));
  }

  ~ColumnFileFdInput() override {}

  bool Next(ColumnFileCompression& compression);

  std::vector<std::pair<uint32_t, kj::Array<const char>>> Fill(
      const std::unordered_set<uint32_t>& field_filter) override;

  bool End() const override { return end_; }

  void SeekToStart() override {
    KJ_REQUIRE(fd_ != nullptr);

    KJ_SYSCALL(lseek(fd_, sizeof(kMagic), SEEK_SET));

    data_.clear();
    buffer_.clear();
    end_ = false;
  }

  // TODO(mortehu): Implement.
  size_t Size() const override { return 0; }

  // TODO(mortehu): Implement.
  size_t Offset() const override { return 0; }

 private:
  struct FieldMeta {
    uint32_t index;
    uint32_t size;
  };

  bool end_ = false;

  std::string buffer_;

  ev::StringRef data_;

  kj::AutoCloseFd fd_;

  std::vector<FieldMeta> field_meta_;

  // Set to true when the file position is at the end of the field data.  This
  // means we have to seek backwards if we want to re-read the data.
  bool at_field_end_ = false;
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

  bool Next(ColumnFileCompression& compression);

  std::vector<std::pair<uint32_t, kj::Array<const char>>> Fill(
      const std::unordered_set<uint32_t>& field_filter) override;

  bool End() const override { return data_.empty(); }

  void SeekToStart() override { data_ = input_data_; }

  size_t Size() const override { return input_data_.size(); }

  size_t Offset() const override { return input_data_.size() - data_.size(); }

 private:
  struct FieldMeta {
    const char* data;
    uint32_t index;
    uint32_t size;
  };

  ev::StringRef input_data_;
  ev::StringRef data_;

  std::vector<FieldMeta> field_meta_;
};

bool ColumnFileFdInput::Next(ColumnFileCompression& compression) {
  KJ_REQUIRE(data_.empty());

  uint8_t size_buffer[4];
  auto ret = Read(fd_, size_buffer, 0, 4);
  if (ret < 4) {
    end_ = true;
    KJ_REQUIRE(ret == 0);
    return false;
  }

  uint32_t size = (size_buffer[0] << 24) | (size_buffer[1] << 16) |
                  (size_buffer[2] << 8) | size_buffer[3];
  try {
    buffer_.resize(size);
  } catch (std::bad_alloc e) {
    KJ_FAIL_REQUIRE("Buffer allocation failed", size);
  }
  Read(fd_, &buffer_[0], size, size);

  data_ = buffer_;

  compression = static_cast<ColumnFileCompression>(GetUInt(data_));

  const auto field_count = GetUInt(data_);

  field_meta_.resize(field_count);

  for (size_t i = 0; i < field_count; ++i) {
    field_meta_[i].index = GetUInt(data_);
    field_meta_[i].size = GetUInt(data_);
  }

  at_field_end_ = false;

  return true;
}

std::vector<std::pair<uint32_t, kj::Array<const char>>> ColumnFileFdInput::Fill(
    const std::unordered_set<uint32_t>& field_filter) {
  std::vector<std::pair<uint32_t, kj::Array<const char>>> result;

  result.reserve(field_filter.empty() ? field_meta_.size()
                                      : field_filter.size());

  if (at_field_end_) {
    off_t reverse_amount = 0;

    for (const auto& f : field_meta_) reverse_amount += f.size;

    KJ_SYSCALL(lseek(fd_, -reverse_amount, SEEK_CUR));
  }

  // Number of bytes to seek before next read.  The purpose of having this
  // variable is to avoid calling lseek several times back-to-back on the
  // same file descriptor.
  size_t skip_amount = 0;

  for (const auto& f : field_meta_) {
    // If the field is ignored, skip its data.
    if (!field_filter.empty() && !field_filter.count(f.index)) {
      skip_amount += f.size;
      continue;
    }

    if (skip_amount > 0) {
      KJ_SYSCALL(lseek(fd_, skip_amount, SEEK_CUR));
      skip_amount = 0;
    }

    auto buffer = kj::heapArray<char>(f.size);
    Read(fd_, buffer.begin(), f.size, f.size);

    result.emplace_back(f.index, std::move(buffer));
  }

  if (skip_amount > 0) {
    KJ_SYSCALL(lseek(fd_, skip_amount, SEEK_CUR));
  }

  at_field_end_ = true;

  return std::move(result);
}

bool ColumnFileStringInput::Next(ColumnFileCompression& compression) {
  KJ_REQUIRE(!data_.empty());

  data_.Consume(4);  // Skip header size we don't need.

  compression = static_cast<ColumnFileCompression>(GetUInt(data_));

  const auto field_count = GetUInt(data_);

  field_meta_.resize(field_count);

  for (size_t i = 0; i < field_count; ++i) {
    field_meta_[i].index = GetUInt(data_);
    field_meta_[i].size = GetUInt(data_);
  }

  for (auto& f : field_meta_) {
    f.data = data_.begin();
    data_.Consume(f.size);
  }

  return true;
}

std::vector<std::pair<uint32_t, kj::Array<const char>>>
ColumnFileStringInput::Fill(const std::unordered_set<uint32_t>& field_filter) {
  std::vector<std::pair<uint32_t, kj::Array<const char>>> result;

  for (const auto& f : field_meta_) {
    if (!field_filter.empty() && !field_filter.count(f.index)) continue;

    // TODO(mortehu): See if we can use a non-owning array instead, e.g.
    //     kj::Array<const char>(data.begin(), f.second,
    //     kj::NullArrayDisposer());
    //  This way, we wouldn't have to copy all the data.

    auto buffer = kj::heapArray<char>(f.data, f.size);

    result.emplace_back(f.index, std::move(buffer));
  }

  return std::move(result);
}

}  // namespace

std::unique_ptr<ColumnFileInput> ColumnFileReader::FileDescriptorInput(
    kj::AutoCloseFd fd) {
  return std::make_unique<ColumnFileFdInput>(std::move(fd));
}

std::unique_ptr<ColumnFileInput> ColumnFileReader::StringInput(
    ev::StringRef data) {
  return std::make_unique<ColumnFileStringInput>(data);
}

ColumnFileReader::ColumnFileReader(std::unique_ptr<ColumnFileInput> input)
    : input_(std::move(input)) {}

ColumnFileReader::ColumnFileReader(kj::AutoCloseFd fd)
    : input_(std::make_unique<ColumnFileFdInput>(std::move(fd))) {}

ColumnFileReader::ColumnFileReader(StringRef input)
    : input_(std::make_unique<ColumnFileStringInput>(input)) {}

ColumnFileReader::ColumnFileReader(ColumnFileReader&&) = default;

ColumnFileReader::~ColumnFileReader() {}

void ColumnFileReader::SetColumnFilter(std::unordered_set<uint32_t> columns) {
  column_filter_ = std::move(columns);
}

bool ColumnFileReader::End() {
  if (!EndOfSegment()) return false;

  if (input_->End()) return true;

  Fill();

  return fields_.empty();
}

bool ColumnFileReader::EndOfSegment() {
  for (auto i = fields_.begin(); i != fields_.end(); i = fields_.erase(i)) {
    if (!i->second.End()) return false;
  }

  return true;
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

const std::vector<std::pair<uint32_t, StringRefOrNull>>&
ColumnFileReader::GetRow() {
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

    if (data) {
      row_buffer_.emplace_back(i->first, *data);
    } else {
      row_buffer_.emplace_back(i->first, nullptr);
    }
  }

  return row_buffer_;
}

void ColumnFileReader::SeekToStart() {
  input_->SeekToStart();

  fields_.clear();
  row_buffer_.clear();
}

void ColumnFileReader::SeekToStartOfSegment() {
  fields_.clear();
  row_buffer_.clear();

  Fill(false);
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

#if HAVE_LZ4
    case kColumnFileCompressionLZ4: {
      ev::StringRef input(data_);
      auto decompressed_size = GetUInt(input);

      auto decompressed_data = kj::heapArray<char>(decompressed_size);
      auto decompress_result =
          LZ4_decompress_safe(input.data(), decompressed_data.begin(),
                              input.size(), decompressed_size);
      KJ_REQUIRE(decompress_result == static_cast<int>(decompressed_size),
                 decompress_result, decompressed_size);

      buffer_ = std::move(decompressed_data);

      data_ = buffer_;
      compression_ = kColumnFileCompressionNone;
    } break;
#endif

#if HAVE_LZMA
    case kColumnFileCompressionLZMA: {
      ev::StringRef input(data_);
      auto decompressed_size = GetUInt(input);

      auto decompressed_data = kj::heapArray<char>(decompressed_size);

      lzma_stream ls = LZMA_STREAM_INIT;

      KJ_REQUIRE(LZMA_OK == lzma_stream_decoder(&ls, UINT64_MAX, 0));

      ls.next_in = reinterpret_cast<const uint8_t*>(input.data());
      ls.avail_in = input.size();
      ls.total_in = input.size();

      ls.next_out = reinterpret_cast<uint8_t*>(decompressed_data.begin());
      ls.avail_out = decompressed_size;

      const auto code_ret = lzma_code(&ls, LZMA_FINISH);
      KJ_REQUIRE(LZMA_STREAM_END == code_ret, code_ret);

      KJ_REQUIRE(ls.total_out == decompressed_size, ls.total_out,
                 decompressed_size);

      buffer_ = std::move(decompressed_data);

      data_ = buffer_;
      compression_ = kColumnFileCompressionNone;
    } break;
#endif

#if HAVE_ZLIB
    case kColumnFileCompressionZLIB: {
      ev::StringRef input(data_);
      auto decompressed_size = GetUInt(input);

      auto decompressed_data = kj::heapArray<char>(decompressed_size);

      z_stream zs;
      memset(&zs, 0, sizeof(zs));

      KJ_REQUIRE(Z_OK == inflateInit(&zs));
      KJ_DEFER(KJ_REQUIRE(Z_OK == inflateEnd(&zs)));

      zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(input.data()));
      zs.avail_in = input.size();
      zs.total_in = input.size();

      zs.next_out = reinterpret_cast<uint8_t*>(decompressed_data.begin());
      zs.avail_out = decompressed_size;

      const auto inflate_ret = inflate(&zs, LZMA_FINISH);
      KJ_REQUIRE(Z_STREAM_END == inflate_ret, inflate_ret, zs.avail_in,
                 zs.total_in, zs.msg);

      KJ_REQUIRE(zs.total_out == decompressed_size, zs.total_out,
                 decompressed_size);

      buffer_ = std::move(decompressed_data);

      data_ = buffer_;
      compression_ = kColumnFileCompressionNone;
    } break;
#endif

    default:
      KJ_FAIL_REQUIRE("Unknown compression scheme", compression_);
  }

  if (!repeat_) {
    repeat_ = GetUInt(data_);

    const auto reserved = GetUInt(data_);
    KJ_REQUIRE(reserved == 0, reserved);

    auto b0 = static_cast<uint8_t>(data_[0]);

    if ((b0 & 0xc0) == 0xc0) {
      data_.Consume(1);
      if (b0 == kCodeNull) {
        value_is_null_ = true;
      } else {
        // The value we're about to read shares a prefix at least 2 bytes long
        // with the previous value.
        const auto shared_prefix = (b0 & 0x3fU) + 2U;
        const auto suffix_length = GetUInt(data_);

        // Verify that the shared prefix isn't longer than the data we've
        // consumed so far.  If it is, the input is corrupt.
        KJ_REQUIRE(shared_prefix <= value_.size(), shared_prefix,
                   value_.size());

        // We just move the old prefix in front of the new suffix, corrupting
        // whatever data is there; we're not going to read it again anyway.
        memmove(const_cast<char*>(data_.data()) - shared_prefix, value_.begin(),
                shared_prefix);

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

void ColumnFileReader::Fill(bool next) {
  fields_.clear();

  if (next && !input_->Next(compression_)) return;

  auto fields = input_->Fill(column_filter_);

  KJ_ASSERT(!fields.empty());

  if (compression_ == kColumnFileCompressionLZMA) {
    if (!thread_pool_) thread_pool_ = std::make_unique<ThreadPool>();

    std::vector<std::pair<uint32_t, std::future<FieldReader>>> future_fields;

    for (auto& field : fields) {
      future_fields.emplace_back(
          field.first, thread_pool_->Launch(
                           [ this, data = std::move(field.second) ]() mutable {
                             FieldReader result(std::move(data), compression_);
                             if (!result.End()) result.Fill();
                             return result;
                           }));
    }

    for (auto& field : future_fields)
      fields_.emplace(field.first, field.second.get());
  } else {
    for (auto& field : fields) {
      fields_.emplace(field.first,
                      FieldReader(std::move(field.second), compression_));
    }
  }
}

}  // namespace ev
