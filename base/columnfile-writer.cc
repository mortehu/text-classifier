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

void PutUInt(std::string& output, uint32_t value) {
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

void PutInt(std::string& output, int32_t value) {
  static const auto sign_shift = sizeof(value) * 8 - 1;

  PutUInt(output, (value << 1) ^ (value >> sign_shift));
}

class ColumnFileFdOutput : public ColumnFileOutput {
 public:
  ColumnFileFdOutput(kj::AutoCloseFd fd) : fd_(std::move(fd)) {
    auto offset = lseek(fd_, 0, SEEK_END);
    if (offset <= 0) WriteAll(fd_, StringRef(kMagic, sizeof(kMagic)));
  }

  void Flush(const std::vector<std::pair<uint32_t, ev::StringRef>>& fields,
             ColumnFileCompression& compression) override;

  kj::AutoCloseFd Finalize() override { return std::move(fd_); }

 private:
  kj::AutoCloseFd fd_;
};

class ColumnFileStringOutput : public ColumnFileOutput {
 public:
  ColumnFileStringOutput(std::string& output) : output_(output) {
    if (output_.empty()) output_.append(kMagic, sizeof(kMagic));
  }

  void Flush(const std::vector<std::pair<uint32_t, ev::StringRef>>& fields,
             ColumnFileCompression& compression) override;

  kj::AutoCloseFd Finalize() override { return nullptr; }

 private:
  std::string& output_;
};

void ColumnFileFdOutput::Flush(
    const std::vector<std::pair<uint32_t, ev::StringRef>>& fields,
    ColumnFileCompression& compression) {
  std::string buffer;
  buffer.resize(4, 0);

  PutUInt(buffer, compression);
  PutUInt(buffer, fields.size());

  for (auto& field : fields) {
    PutUInt(buffer, field.first);
    PutUInt(buffer, field.second.size());
  }

  auto buffer_size = buffer.size() - 4;  // Don't count the size itself.
  buffer[0] = buffer_size >> 24U;
  buffer[1] = buffer_size >> 16U;
  buffer[2] = buffer_size >> 8U;
  buffer[3] = buffer_size;

  WriteAll(fd_, buffer);

  for (const auto& field : fields) WriteAll(fd_, field.second);
}

void ColumnFileStringOutput::Flush(
    const std::vector<std::pair<uint32_t, ev::StringRef>>& fields,
    ColumnFileCompression& compression) {
  std::string buffer;
  buffer.resize(4, 0);

  PutUInt(buffer, compression);
  PutUInt(buffer, fields.size());

  for (auto& field : fields) {
    PutUInt(buffer, field.first);
    PutUInt(buffer, field.second.size());
  }

  auto buffer_size = buffer.size() - 4;  // Don't count the size itself.
  buffer[0] = buffer_size >> 24U;
  buffer[1] = buffer_size >> 16U;
  buffer[2] = buffer_size >> 8U;
  buffer[3] = buffer_size;

  output_ += buffer;

  for (const auto& field : fields)
    output_.append(field.second.begin(), field.second.end());
}

}  // namespace

ColumnFileWriter::ColumnFileWriter(std::unique_ptr<ColumnFileOutput> output) {}

ColumnFileWriter::ColumnFileWriter(kj::AutoCloseFd&& fd)
    : output_(std::make_unique<ColumnFileFdOutput>(std::move(fd))) {}

ColumnFileWriter::ColumnFileWriter(const char* path, int mode)
    : output_(std::make_unique<ColumnFileFdOutput>(
          OpenFile(path, O_CREAT | O_WRONLY, mode))) {}

ColumnFileWriter::ColumnFileWriter(std::string& output)
    : output_(std::make_unique<ColumnFileStringOutput>(output)) {}

ColumnFileWriter::~ColumnFileWriter() { Finalize(); }

void ColumnFileWriter::Put(uint32_t column, const StringRef& data) {
  fields_[column].Put(data);
}

void ColumnFileWriter::PutNull(uint32_t column) { fields_[column].PutNull(); }

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

    field_it->second.Put(row_it->second);

    ++row_it;
    ++field_it;
  }
}

void ColumnFileWriter::Flush() {
  if (fields_.empty()) return;

  std::vector<std::pair<uint32_t, ev::StringRef>> field_data;
  field_data.reserve(fields_.size());

  for (auto& field : fields_) {
    field.second.Finalize(compression_);
    field_data.emplace_back(field.first, field.second.Data());
  }

  output_->Flush(field_data, compression_);

  fields_.clear();
}

kj::AutoCloseFd ColumnFileWriter::Finalize() {
  Flush();
  return output_->Finalize();
}

void ColumnFileWriter::FieldWriter::Put(const StringRef& data) {
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

  if (data_mismatch) {
    Flush();
    if (data_mismatch) {
      value_.assign(data.begin(), data.end());
      value_is_null_ = false;
      shared_prefix_ = shared_prefix;
    }
  }

  ++repeat_;
}

void ColumnFileWriter::FieldWriter::PutNull() {
  if (!value_is_null_) Flush();

  value_is_null_ = true;
  ++repeat_;
}

void ColumnFileWriter::FieldWriter::Flush() {
  if (!repeat_) return;

  PutUInt(data_, repeat_);
  PutUInt(data_, 0);  // Reserved field.

  if (value_is_null_) {
    data_.push_back(kCodeNull);
  } else {
    if (shared_prefix_ > 2) {
      // Make sure we don't produce 0xff in the output, which is used to
      // indicate NULL values.
      if (shared_prefix_ > 0x40) shared_prefix_ = 0x40;
      data_.push_back(0xc0 | (shared_prefix_ - 2));
      PutUInt(data_, value_.size() - shared_prefix_);
      data_.append(value_.begin() + shared_prefix_, value_.end());
    } else {
      PutUInt(data_, value_.size());
      data_.append(value_.begin(), value_.end());
    }
  }

  repeat_ = 0;
  value_is_null_ = true;
}

void ColumnFileWriter::FieldWriter::Finalize(
    ColumnFileCompression compression) {
  Flush();

  switch (compression) {
    case kColumnFileCompressionNone:
      break;

#if HAVE_LIBSNAPPY
    case kColumnFileCompressionSnappy: {
      std::string compressed_data;
      compressed_data.resize(snappy::MaxCompressedLength(data_.size()));
      size_t compressed_length = SIZE_MAX;
      snappy::RawCompress(data_.data(), data_.size(), &compressed_data[0],
                          &compressed_length);
      KJ_REQUIRE(compressed_length <= compressed_data.size());
      compressed_data.resize(compressed_length);
      data_.swap(compressed_data);
      KJ_REQUIRE(snappy::IsValidCompressedBuffer(data_.data(), data_.size()));
    } break;
#endif

    default:
      KJ_FAIL_REQUIRE("Unsupported compression scheme", compression);
  }
}

}  // namespace ev
