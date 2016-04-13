#include "base/columnfile.h"

#include <fcntl.h>
#include <unistd.h>

#include <kj/array.h>
#include <kj/debug.h>
#include <lz4.h>
#include <lzma.h>
#include <snappy.h>

#include "base/columnfile-internal.h"
#include "base/compression.h"
#include "base/file.h"
#include "base/macros.h"
#include "base/thread-pool.h"

namespace ev {

namespace {

using namespace columnfile_internal;

class ColumnFileFdOutput : public ColumnFileOutput {
 public:
  ColumnFileFdOutput(kj::AutoCloseFd fd) : fd_(std::move(fd)) {
    auto offset = lseek(fd_, 0, SEEK_END);
    if (offset <= 0) WriteAll(fd_, StringRef(kMagic, sizeof(kMagic)));
  }

  void Flush(const std::vector<std::pair<uint32_t, ev::StringRef>>& fields,
             const ColumnFileCompression compression) override;

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
             const ColumnFileCompression compression) override;

  kj::AutoCloseFd Finalize() override { return nullptr; }

 private:
  std::string& output_;
};

void ColumnFileFdOutput::Flush(
    const std::vector<std::pair<uint32_t, ev::StringRef>>& fields,
    const ColumnFileCompression compression) {
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
    const ColumnFileCompression compression) {
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

ColumnFileCompression ColumnFileWriter::StringToCompressingAlgorithm(
    const ev::StringRef& name) {
  if (name == "none") return kColumnFileCompressionNone;
#if HAVE_LIBSNAPPY
  if (name == "snappy") return kColumnFileCompressionSnappy;
#endif
#if HAVE_LZ4
  if (name == "lz4") return kColumnFileCompressionLZ4;
#endif
#if HAVE_LZMA
  if (name == "lzma") return kColumnFileCompressionLZMA;
#endif
#if HAVE_ZLIB
  if (name == "zlib") return kColumnFileCompressionZLIB;
#endif
  KJ_FAIL_REQUIRE("Unsupported compression algorithm", name);
}

ColumnFileWriter::ColumnFileWriter(std::shared_ptr<ColumnFileOutput> output)
    : output_(std::move(output)) {}

ColumnFileWriter::ColumnFileWriter(kj::AutoCloseFd&& fd)
    : output_(std::make_shared<ColumnFileFdOutput>(std::move(fd))) {}

ColumnFileWriter::ColumnFileWriter(const char* path, int mode)
    : output_(std::make_shared<ColumnFileFdOutput>(
          OpenFile(path, O_CREAT | O_WRONLY, mode))) {}

ColumnFileWriter::ColumnFileWriter(std::string& output)
    : output_(std::make_shared<ColumnFileStringOutput>(output)) {}

ColumnFileWriter::~ColumnFileWriter() { Finalize(); }

void ColumnFileWriter::Put(uint32_t column, const StringRef& data) {
  fields_[column].Put(data);
  pending_size_ += data.size();
}

void ColumnFileWriter::PutNull(uint32_t column) {
  fields_[column].PutNull();
  ++pending_size_;
}

void ColumnFileWriter::PutRow(
    const std::vector<std::pair<uint32_t, StringRefOrNull>>& row) {
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

    if (row_it->second.IsNull()) {
      field_it->second.PutNull();
    } else {
      const auto& str = row_it->second.StringRef();
      field_it->second.Put(str);
      pending_size_ += str.size();
    }

    ++row_it;
    ++field_it;
  }
}

void ColumnFileWriter::Flush() {
  if (fields_.empty()) return;

  if (!thread_pool_) thread_pool_ = std::make_unique<ThreadPool>();

  std::vector<std::pair<uint32_t, ev::StringRef>> field_data;
  field_data.reserve(fields_.size());

  for (auto& field : fields_) {
    field.second.Finalize(compression_, *thread_pool_.get());
    field_data.emplace_back(field.first, field.second.Data());
  }

  output_->Flush(field_data, compression_);

  fields_.clear();

  pending_size_ = 0;
}

kj::AutoCloseFd ColumnFileWriter::Finalize() {
  if (!output_) return nullptr;
  Flush();
  auto result = output_->Finalize();
  output_.reset();
  return result;
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

void ColumnFileWriter::FieldWriter::Finalize(ColumnFileCompression compression,
                                             ev::ThreadPool& thread_pool) {
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
    } break;
#endif

#if HAVE_LZ4
    case kColumnFileCompressionLZ4: {
      std::string compressed_data;
      PutUInt(compressed_data, data_.size());
      const auto data_offset = compressed_data.size();
      compressed_data.resize(data_offset + LZ4_compressBound(data_.size()));

      const auto compressed_length = LZ4_compress(
          data_.data(), &compressed_data[data_offset], data_.size());
      KJ_REQUIRE(data_offset + compressed_length <= compressed_data.size());
      compressed_data.resize(data_offset + compressed_length);
      data_.swap(compressed_data);
    } break;
#endif

#if HAVE_LZMA
    case kColumnFileCompressionLZMA: {
      std::string compressed_data;
      PutUInt(compressed_data, data_.size());
      const auto data_offset = compressed_data.size();
      compressed_data.resize(data_offset +
                             lzma_stream_buffer_bound(data_.size()));

      lzma_stream ls = LZMA_STREAM_INIT;

      KJ_REQUIRE(LZMA_OK == lzma_easy_encoder(&ls, 1, LZMA_CHECK_CRC32));

      ls.next_in = reinterpret_cast<const uint8_t*>(data_.data());
      ls.avail_in = data_.size();
      ls.total_in = data_.size();

      ls.next_out = reinterpret_cast<uint8_t*>(&compressed_data[data_offset]);
      ls.avail_out = compressed_data.size() - data_offset;

      const auto code_ret = lzma_code(&ls, LZMA_FINISH);
      KJ_REQUIRE(LZMA_STREAM_END == code_ret, code_ret);

      const auto compressed_length = ls.total_out;
      KJ_REQUIRE(data_offset + compressed_length <= compressed_data.size());

      lzma_end(&ls);

      compressed_data.resize(data_offset + compressed_length);
      data_.swap(compressed_data);
    } break;
#endif

#if HAVE_ZLIB
    case kColumnFileCompressionZLIB: {
      std::string compressed_data;
      PutUInt(compressed_data, data_.size());

      CompressZLIB(compressed_data, data_, thread_pool);

      data_.swap(compressed_data);
    } break;
#endif

    default:
      KJ_FAIL_REQUIRE("Unknown compression scheme", compression);
  }
}

}  // namespace ev
