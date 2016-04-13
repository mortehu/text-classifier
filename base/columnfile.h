#ifndef BASE_COLUMNFILE_H_
#define BASE_COLUMNFILE_H_ 1

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_set>

#include <kj/debug.h>
#include <kj/io.h>

#include "base/region.h"
#include "base/stringref.h"

namespace ev {

class ThreadPool;

enum ColumnFileCompression : uint32_t {
  kColumnFileCompressionNone = 0,

#if HAVE_LIBSNAPPY
  kColumnFileCompressionSnappy = 1,
#endif

#if HAVE_LZ4
  kColumnFileCompressionLZ4 = 2,
#endif

#if HAVE_LZMA
  kColumnFileCompressionLZMA = 3,
#endif

#if HAVE_ZLIB
  kColumnFileCompressionZLIB = 4,
#endif

#if HAVE_LZ4
  kColumnFileCompressionDefault = kColumnFileCompressionLZ4
#elif HAVE_LIBSNAPPY
  kColumnFileCompressionDefault = kColumnFileCompressionSnappy
#else
  kColumnFileCompressionDefault = kColumnFileCompressionNone
#endif
};

class ColumnFileOutput {
 public:
  virtual ~ColumnFileOutput() noexcept(false) {}

  virtual void Flush(
      const std::vector<std::pair<uint32_t, ev::StringRef>>& fields,
      const ColumnFileCompression compression) = 0;

  // Finishes writing the file.  Returns the underlying file descriptor, if
  // available.
  virtual kj::AutoCloseFd Finalize() = 0;
};

class ColumnFileWriter {
 public:
  static ColumnFileCompression StringToCompressingAlgorithm(
      const ev::StringRef& name);

  explicit ColumnFileWriter(std::shared_ptr<ColumnFileOutput> output);

  explicit ColumnFileWriter(kj::AutoCloseFd&& fd);

  ColumnFileWriter(const char* path, int mode = 0666);

  explicit ColumnFileWriter(std::string& output);

  ~ColumnFileWriter();

  void SetCompression(ColumnFileCompression c) { compression_ = c; }

  // Inserts a value.
  void Put(uint32_t column, const StringRef& data);
  void PutNull(uint32_t column);

  void PutRow(const std::vector<std::pair<uint32_t, StringRefOrNull>>& row);

  // Returns an approximate number of uncompressed bytes that have not yet been
  // flushed.  This can be used to make a decision as to whether or not to call
  // `Flush()`.
  size_t PendingSize() const { return pending_size_; }

  // Writes all buffered records to the output stream.
  void Flush();

  // Finishes writing the file.  Returns the underlying file descriptor.
  //
  // This function is implicitly called by the destructor.
  kj::AutoCloseFd Finalize();

 private:
  class FieldWriter {
   public:
    void Put(const StringRef& data);

    void PutNull();

    void Flush();

    void Finalize(ColumnFileCompression compression,
                  ev::ThreadPool& thread_pool);

    StringRef Data() const { return data_; }

   private:
    std::string data_;

    std::string value_;
    bool value_is_null_ = false;

    uint32_t repeat_ = 0;

    unsigned int shared_prefix_ = 0;
  };

  std::unique_ptr<ev::ThreadPool> thread_pool_;

  std::shared_ptr<ColumnFileOutput> output_;

  ColumnFileCompression compression_ = kColumnFileCompressionLZ4;

  std::map<uint32_t, FieldWriter> fields_;

  size_t pending_size_ = 0;
};

class ColumnFileInput {
 public:
  virtual ~ColumnFileInput() noexcept(false) {}

  // Moves to the next segment.  Returns `false` if the end of the input stream
  // was reached, `true` otherwise.
  virtual bool Next(ColumnFileCompression& compression) = 0;

  // Returns the data chunks for the fields specified in `field_filter`.  If
  // `field_filter` is empty, all fields are selected.
  virtual std::vector<std::pair<uint32_t, kj::Array<const char>>> Fill(
      const std::unordered_set<uint32_t>& field_filter) = 0;

  // Returns `true` if the next call to `Fill` will definitely return an
  // empty vector, `false` otherwise.
  virtual bool End() const = 0;

  // Seek to the beginning of the input.
  virtual void SeekToStart() = 0;

  // Returns the size of the input, in an unspecified unit.
  virtual size_t Size() const = 0;

  // Returns the approximate offset, in an unspecified unit.  This value only
  // makes sense when compared to the return value of `Size()`.
  virtual size_t Offset() const = 0;
};

class ColumnFileReader {
 public:
  static std::unique_ptr<ColumnFileInput> FileDescriptorInput(
      kj::AutoCloseFd fd);

  static std::unique_ptr<ColumnFileInput> StringInput(ev::StringRef data);

  explicit ColumnFileReader(std::unique_ptr<ColumnFileInput> input);

  // Reads a column file as a stream.  If you want to use memory-mapped I/O,
  // use the StringRef based constructor below.
  explicit ColumnFileReader(kj::AutoCloseFd fd);

  // Reads a column file from memory.
  explicit ColumnFileReader(StringRef input);

  ColumnFileReader(ColumnFileReader&&);

  ColumnFileReader& operator=(ColumnFileReader&&) = default;

  KJ_DISALLOW_COPY(ColumnFileReader);

  ~ColumnFileReader();

  void SetColumnFilter(std::unordered_set<uint32_t> columns);

  template <typename Iterator>
  void SetColumnFilter(Iterator begin, Iterator end) {
    column_filter_.clear();
    while (begin != end) column_filter_.emplace(*begin++);
  }

  // Returns true iff there's no more data to be read.
  bool End();

  // Returns true iff there's no more data to be read in the current segment.
  bool EndOfSegment();

  const StringRef* Peek(uint32_t field);
  const StringRef* Get(uint32_t field);

  const std::vector<std::pair<uint32_t, StringRefOrNull>>& GetRow();

  void SeekToStart();

  void SeekToStartOfSegment();

  size_t Size() const { return input_->Size(); }

  size_t Offset() const { return input_->Offset(); }

 private:
  class FieldReader {
   public:
    FieldReader(kj::Array<const char> buffer,
                ColumnFileCompression compression);

    FieldReader(FieldReader&&) = default;
    FieldReader& operator=(FieldReader&&) = default;

    KJ_DISALLOW_COPY(FieldReader);

    bool End() const { return !repeat_ && data_.empty(); }

    const StringRef* Peek() {
      if (!repeat_) {
        KJ_ASSERT(!data_.empty());
        Fill();
        KJ_ASSERT(repeat_ > 0);
      }

      return value_is_null_ ? nullptr : &value_;
    }

    const StringRef* Get() {
      auto result = Peek();
      --repeat_;
      return result;
    }

    void Fill();

   private:
    kj::Array<const char> buffer_;

    StringRef data_;

    ColumnFileCompression compression_;

    StringRef value_;
    bool value_is_null_ = true;
    uint32_t array_size_ = 0;

    uint32_t repeat_ = 0;
  };

  void Fill(bool next = true);

  std::unique_ptr<ev::ThreadPool> thread_pool_;

  std::unique_ptr<ColumnFileInput> input_;

  std::unordered_set<uint32_t> column_filter_;

  ColumnFileCompression compression_;

  std::map<uint32_t, FieldReader> fields_;

  std::vector<std::pair<uint32_t, StringRefOrNull>> row_buffer_;
};

class ColumnFileSelect {
 public:
  ColumnFileSelect(ColumnFileReader input);

  void AddSelection(uint32_t field);

  void AddFilter(uint32_t field,
                 std::function<bool(const StringRefOrNull&)> filter);

  void StartScan();

  kj::Maybe<const std::vector<std::pair<uint32_t, StringRefOrNull>>&> Iterate(
      ev::concurrency::RegionPool& region_pool);

  // Convenience wrapper for `StartScan()` and `Iterate()`.
  void Execute(ev::concurrency::RegionPool& region_pool,
               std::function<void(const std::vector<
                   std::pair<uint32_t, StringRefOrNull>>&)> callback);

 private:
  struct RowCache {
    std::vector<std::pair<uint32_t, StringRefOrNull>> data;
    uint32_t index;
  };

  // Reads the next chunk into `row_buffer_`, removing any rows not matching
  // the filters in `filters_`.
  void ReadChunk(ev::concurrency::RegionPool& region_pool);

  ColumnFileReader input_;

  std::unordered_set<uint32_t> selection_;

  std::vector<std::pair<uint32_t, std::function<bool(const StringRefOrNull&)>>>
      filters_;

  ev::concurrency::RegionPool::Region region_;

  std::vector<RowCache> row_buffer_;
  size_t chunk_offset_ = 0;

  // Columns that appear in `selection_`, but not in `filters_`.
  std::unordered_set<uint32_t> unfiltered_selection_;
};

}  // namespace ev

#endif  // !BASE_COLUMNFILE_H_
