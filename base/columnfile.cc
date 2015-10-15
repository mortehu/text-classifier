#include "base/columnfile.h"

#include <fcntl.h>
#include <unistd.h>

#include <kj/debug.h>
#include <snappy.h>

#include "base/file.h"
#include "base/macros.h"

namespace ev {

namespace {

// The magic code string is designed to cause a parse error if someone attempts
// to parse the file as a CSV.
const char kMagic[4] = {'\n', '\t', '\"', 0};

enum Codes : uint8_t {
  kCodeNull = 0xff,
};

}  // namespace

void ColumnFileWriter::PutInt(std::string& output, uint32_t value) {
  if (value < (1 << 7)) {
    output.push_back(value);
  } else if (value < (1 << 13)) {
    output.push_back((value & 0x3f) | 0x80);
    output.push_back(value >> 6);
  } else if (value < (1 << 20)) {
    output.push_back((value & 0x3f) | 0x80);
    output.push_back((value >> 6) | 0x80);
    output.push_back(value >> 13);
  } else if (value < (1 << 27)) {
    output.push_back((value & 0x3f) | 0x80);
    output.push_back((value >> 6) | 0x80);
    output.push_back((value >> 13) | 0x80);
    output.push_back(value >> 20);
  } else {
    output.push_back((value & 0x3f) | 0x80);
    output.push_back((value >> 6) | 0x80);
    output.push_back((value >> 13) | 0x80);
    output.push_back((value >> 20) | 0x80);
    output.push_back(value >> 27);
  }
}

uint32_t ColumnFileReader::GetInt(StringRef& input) {
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

ColumnFileWriter::ColumnFileWriter(kj::AutoCloseFd&& fd) : fd_(std::move(fd)) {
  auto offset = lseek(fd_, 0, SEEK_END);
  if (offset <= 0) WriteAll(fd_, StringRef(kMagic, sizeof(kMagic)));
}

ColumnFileWriter::ColumnFileWriter(const char* path, int mode)
    : ColumnFileWriter(OpenFile(path, O_CREAT | O_WRONLY, mode)) {}

ColumnFileWriter::ColumnFileWriter(std::string& output)
    : output_string_(&output) {
  if (output.empty()) output_string_->append(kMagic, sizeof(kMagic));
}

ColumnFileWriter::~ColumnFileWriter() { Finalize(); }

void ColumnFileWriter::Put(uint8_t level, uint32_t column,
                           const StringRef& data) {
  fields_[column].Put(level, data);
}

void ColumnFileWriter::PutNull(uint8_t level, uint32_t column) {
  fields_[column].PutNull(level);
}

void ColumnFileWriter::PutRow(
    const std::vector<std::pair<uint32_t, StringRef>>& row) {
  // We iterate simultaneously through the fields_ map and the row, so that if
  // their keys matches, we don't have to perform any binary searches in the
  // map.
  auto field_it = fields_.begin();
  auto row_it = row.begin();

  while (row_it != row.end()) {
    if (EV_UNLIKELY(field_it == fields_.end() ||
                    field_it->first != row_it->first)) {
      field_it = fields_.find(row_it->first);
      if (field_it == fields_.end())
        field_it = fields_.emplace(row_it->first, FieldWriter()).first;
    }

    field_it->second.Put(0, row_it->second);

    ++row_it;
    ++field_it;
  }
}

void ColumnFileWriter::Flush() {
  if (fields_.empty()) return;

  std::string buffer;
  buffer.resize(4, 0);

  PutInt(buffer, compression_);
  PutInt(buffer, fields_.size());

  for (auto& field : fields_) {
    field.second.Finalize(compression_);

    PutInt(buffer, field.first);
    PutInt(buffer, field.second.Data().size());
  }

  auto buffer_size = buffer.size() - 4;  // Don't count the size itself.
  buffer[0] = buffer_size >> 24U;
  buffer[1] = buffer_size >> 16U;
  buffer[2] = buffer_size >> 8U;
  buffer[3] = buffer_size;

  if (output_string_)
    output_string_->append(buffer.begin(), buffer.end());
  else
    WriteAll(fd_, buffer);

  for (const auto& field : fields_) {
    auto field_buffer = field.second.Data();

    if (output_string_)
      output_string_->append(field_buffer.begin(), field_buffer.end());
    else
      WriteAll(fd_, field_buffer);
  }

  fields_.clear();
}

kj::AutoCloseFd ColumnFileWriter::Finalize() {
  Flush();
  return std::move(fd_);
}

void ColumnFileWriter::FieldWriter::Put(uint8_t level, const StringRef& data) {
  bool data_mismatch;
  unsigned int shared_prefix = 0;
  if (value_is_null_) {
    data_mismatch = true;
  } else {
    auto i =
        std::mismatch(data.begin(), data.end(), value_.begin(), value_.end());
    if (i.first != data.end() || i.second != value_.end()) {
      shared_prefix = std::distance(data.begin(), i.first);
      data_mismatch = true;
    } else {
      data_mismatch = false;
    }
  }

  if (level != level_ || data_mismatch) {
    Flush();
    if (data_mismatch) {
      value_.assign(data.begin(), data.end());
      value_is_null_ = false;
      shared_prefix_ = shared_prefix;
    }
  }

  level_ = level;
  ++repeat_;
}

void ColumnFileWriter::FieldWriter::PutNull(uint8_t level) {
  if (level != level_ || !value_is_null_) Flush();

  level_ = level;
  value_is_null_ = true;
  ++repeat_;
}

void ColumnFileWriter::FieldWriter::Flush() {
  if (!repeat_) return;

  PutInt(data_, repeat_);
  data_.push_back(level_);

  if (value_is_null_) {
    data_.push_back(kCodeNull);
  } else {
    if (shared_prefix_ > 2) {
      // Make sure we don't produce 0xff in the output, which is used to
      // indicate NULL values.
      if (shared_prefix_ > 0x40) shared_prefix_ = 0x40;
      data_.push_back(0xc0 | (shared_prefix_ - 2));
      PutInt(data_, value_.size() - shared_prefix_);
      data_.append(value_.begin() + shared_prefix_, value_.end());
    } else {
      PutInt(data_, value_.size());
      data_.append(value_.begin(), value_.end());
    }
  }

  repeat_ = 0;
  value_is_null_ = true;
}

void ColumnFileWriter::FieldWriter::Finalize(
    ColumnFileCompression compression) {
  Flush();

  if (compression == kColumnFileCompressionSnappy) {
    std::string compressed_data;
    compressed_data.resize(snappy::MaxCompressedLength(data_.size()));
    size_t compressed_length = SIZE_MAX;
    snappy::RawCompress(data_.data(), data_.size(), &compressed_data[0],
                        &compressed_length);
    KJ_REQUIRE(compressed_length <= compressed_data.size());
    compressed_data.resize(compressed_length);
    data_.swap(compressed_data);
    KJ_REQUIRE(snappy::IsValidCompressedBuffer(data_.data(), data_.size()));
  }
}

ColumnFileReader::ColumnFileReader(kj::AutoCloseFd&& fd) : fd_(std::move(fd)) {
  (void)posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);

  char magic[sizeof(kMagic)];
  Read(fd_, magic, sizeof(kMagic), sizeof(kMagic));
  KJ_REQUIRE(!memcmp(magic, kMagic, sizeof(kMagic)));
}

ColumnFileReader::ColumnFileReader(StringRef input) : data_(std::move(input)) {
  KJ_REQUIRE(data_.size() >= sizeof(kMagic));
  KJ_REQUIRE(!memcmp(data_.begin(), kMagic, sizeof(kMagic)));
  data_.Consume(sizeof(kMagic));
}

void ColumnFileReader::SetColumnFilter(
    std::initializer_list<uint32_t> columns) {
  column_filter_.clear();
  for (auto column : columns) column_filter_.emplace(column);
}

bool ColumnFileReader::End() {
  for (auto i = fields_.begin(); i != fields_.end(); i = fields_.erase(i))
    if (!i->second.End()) return false;

  if (data_.empty() && fd_ != nullptr) Fill();

  return data_.empty() && fields_.empty();
}

const std::vector<std::pair<uint32_t, StringRef>>& ColumnFileReader::GetRow() {
  row_buffer_.clear();

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
    auto next_level = i->second.NextLevel();

    // We don't support nested values when reading a flat row.
    KJ_REQUIRE(next_level == 0, next_level);

    auto value = i->second.Get();
    if (value) row_buffer_.emplace_back(i->first, *value);
  }

  return row_buffer_;
}

void ColumnFileReader::SeekToStart() {
  // TODO(mortehu): Add support for seeking in buffers.
  KJ_REQUIRE(fd_ != nullptr);

  KJ_SYSCALL(lseek(fd_, sizeof(kMagic), SEEK_SET));

  buffer_.clear();
  data_.clear();
  fields_.clear();
  row_buffer_.clear();
}

ColumnFileReader::FieldReader::FieldReader(std::unique_ptr<char[]> buffer,
                                           size_t buffer_size,
                                           ColumnFileCompression compression)
    : buffer_(std::move(buffer)),
      buffer_size_(buffer_size),
      data_(buffer_.get(), buffer_size_),
      compression_(compression) {}

ColumnFileReader::FieldReader::FieldReader(StringRef data,
                                           ColumnFileCompression compression)
    : data_(std::move(data)), compression_(compression) {}

void ColumnFileReader::FieldReader::Fill() {
  if (compression_ != kColumnFileCompressionNone) {
    KJ_REQUIRE(compression_ == kColumnFileCompressionSnappy);
    KJ_REQUIRE(snappy::IsValidCompressedBuffer(data_.data(), data_.size()));
    size_t decompressed_size = 0;
    KJ_REQUIRE(snappy::GetUncompressedLength(data_.data(), data_.size(),
                                             &decompressed_size));

    std::unique_ptr<char[]> decompressed_data(new char[decompressed_size]);
    KJ_REQUIRE(snappy::RawUncompress(data_.data(), data_.size(),
                                     decompressed_data.get()));
    buffer_.swap(decompressed_data);
    buffer_size_ = decompressed_size;

    data_ = ev::StringRef(buffer_.get(), buffer_size_);
    compression_ = kColumnFileCompressionNone;
  }

  if (!repeat_) {
    repeat_ = GetInt(data_);
    level_ = data_[0];
    data_.Consume(1);

    auto b0 = static_cast<uint8_t>(data_[0]);

    if ((b0 & 0xc0) == 0xc0) {
      data_.Consume(1);
      if (b0 == 0xff) {
        value_is_null_ = true;
      } else {
        // The value we're about to read shares a prefix at least 2 bytes long
        // with the previous value.
        auto shared_prefix = (b0 & 0x3f) + 2;
        auto suffix_length = GetInt(data_);

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
      auto value_size = GetInt(data_);
      value_ = StringRef(data_.begin(), value_size);
      data_.Consume(value_size);
      value_is_null_ = false;
    }
  }
}

void ColumnFileReader::Fill() {
  if (fd_ != nullptr) {
    KJ_REQUIRE(data_.empty());
    uint8_t size_buffer[4];
    auto ret = Read(fd_, size_buffer, 0, 4);
    if (ret < 4) {
      KJ_REQUIRE(ret == 0);
      return;
    }

    auto size = (size_buffer[0] << 24) | (size_buffer[1] << 16) |
                (size_buffer[2] << 8) | size_buffer[3];
    buffer_.resize(size);
    Read(fd_, &buffer_[0], size, size);
    data_ = buffer_;
  } else {
    KJ_REQUIRE(!data_.empty());
    data_.Consume(4);  // Skip header size we don't need.
  }

  std::vector<std::pair<uint32_t, uint32_t>> field_sizes;

  ColumnFileCompression compression =
      static_cast<ColumnFileCompression>(GetInt(data_));

  fields_.clear();
  auto field_count = GetInt(data_);
  for (auto i = field_count; i-- > 0;) {
    auto field_idx = GetInt(data_);
    auto field_size = GetInt(data_);
    field_sizes.emplace_back(field_idx, field_size);
  }

  if (fd_ != nullptr) {
    // Number of bytes to seek before next read.  The purpose of having this
    // variable is to avoid calling lseek several times back-to-back on the
    // same file descriptor.
    size_t seek_amount = 0;

    for (const auto& f : field_sizes) {
      // If the column is ignored, skip its data.
      if (!column_filter_.empty() && !column_filter_.count(f.first)) {
        seek_amount += f.second;
        continue;
      }

      if (seek_amount > 0) {
        KJ_SYSCALL(lseek(fd_, seek_amount, SEEK_CUR));
        seek_amount = 0;
      }

      std::unique_ptr<char[]> buffer(new char[f.second]);
      Read(fd_, buffer.get(), f.second, f.second);

      fields_.emplace_back(
          f.first, FieldReader(std::move(buffer), f.second, compression));
    }

    if (seek_amount > 0) {
      KJ_SYSCALL(lseek(fd_, seek_amount, SEEK_CUR));
    }
  } else {
    for (const auto& f : field_sizes) {
      // If the column is ignored, skip its data.
      if (!column_filter_.empty() && !column_filter_.count(f.first)) {
        data_.Consume(f.second);
        continue;
      }

      fields_.emplace_back(
          f.first,
          FieldReader(StringRef(data_.begin(), f.second), compression));
      data_.Consume(f.second);
    }
  }
}

}  // namespace ev
