#ifndef BASE_COLUMNFILE_H_
#define BASE_COLUMNFILE_H_ 1

#include <cstdint>
#include <map>
#include <unordered_set>

#include <kj/debug.h>
#include <kj/io.h>

#include "base/stringref.h"

namespace ev {

enum ColumnFileCompression : uint32_t {
  kColumnFileCompressionNone = 0,
  kColumnFileCompressionSnappy = 1,
};

class ColumnFileOutput {
 public:
  virtual ~ColumnFileOutput() noexcept(false) {}

  virtual void Flush(
      const std::vector<std::pair<uint32_t, ev::StringRef>>& fields,
      ColumnFileCompression& compression) = 0;

  // Finishes writing the file.  Returns the underlying file descriptor, if
  // available.
  virtual kj::AutoCloseFd Finalize() = 0;
};

class ColumnFileWriter {
 public:
  ColumnFileWriter(std::unique_ptr<ColumnFileOutput> output);

  ColumnFileWriter(kj::AutoCloseFd&& fd);

  ColumnFileWriter(const char* path, int mode = 0666);

  ColumnFileWriter(std::string& output);

  ~ColumnFileWriter();

  void SetCompression(ColumnFileCompression c) { compression_ = c; }

  // Inserts a value.
  void Put(uint32_t column, const StringRef& data);
  void PutNull(uint32_t column);

  void PutRow(const std::vector<std::pair<uint32_t, StringRef>>& row);

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

    void Finalize(ColumnFileCompression compression);

    StringRef Data() const { return data_; }

   private:
    std::string data_;

    std::string value_;
    bool value_is_null_ = false;

    uint32_t repeat_ = 0;

    unsigned int shared_prefix_ = 0;
  };

  std::unique_ptr<ColumnFileOutput> output_;

  ColumnFileCompression compression_ = kColumnFileCompressionSnappy;

  std::map<uint32_t, FieldWriter> fields_;
};

class ColumnFileInput {
 public:
  virtual ~ColumnFileInput() noexcept(false) {}

  // Returns the next data chunks for all fields.  If `field_filter` is not
  // empty, only fields specified in this set are included.
  //
  // Returns an empty vector if the end was reached.
  virtual std::vector<std::pair<uint32_t, kj::Array<const char>>> Fill(
      const std::unordered_set<uint32_t>& field_filter,
      ColumnFileCompression& compression) = 0;

  // Returns `true` if the next call to `Fill` will definitely return an
  // empty vector, `false` otherwise.
  virtual bool End() const = 0;

  virtual void SeekToStart() = 0;
};

class ColumnFileReader {
 public:
  ColumnFileReader(std::unique_ptr<ColumnFileInput> input);

  // Reads a column file as a stream.  If you want to use memory-mapped I/O,
  // use the StringRef based constructor below.
  ColumnFileReader(kj::AutoCloseFd fd);

  // Reads a column file from memory, or virtual address space.
  ColumnFileReader(StringRef input);

  ColumnFileReader(ColumnFileReader&&) = default;

  ColumnFileReader& operator=(ColumnFileReader&&) = default;

  KJ_DISALLOW_COPY(ColumnFileReader);

  void SetColumnFilter(std::initializer_list<uint32_t> columns);

  template <typename Iterator>
  void SetColumnFilter(Iterator begin, Iterator end) {
    column_filter_.clear();
    while (begin != end) column_filter_.emplace(*begin++);
  }

  // Returns true iff there's no more data to be read.
  bool End();

  const StringRef* Peek(uint32_t field);
  const StringRef* Get(uint32_t field);

  const std::vector<std::pair<uint32_t, StringRef>>& GetRow();

  void SeekToStart();

 private:
  class FieldReader {
   public:
    FieldReader(kj::Array<const char> buffer, ColumnFileCompression compression);

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

   private:
    void Fill();

    kj::Array<const char> buffer_;

    StringRef data_;

    ColumnFileCompression compression_;

    StringRef value_;
    bool value_is_null_ = true;
    uint32_t array_size_ = 0;

    uint32_t repeat_ = 0;
  };

  void Fill();

  std::unique_ptr<ColumnFileInput> input_;

  std::unordered_set<uint32_t> column_filter_;

  std::map<uint32_t, FieldReader> fields_;

  std::vector<std::pair<uint32_t, StringRef>> row_buffer_;
};

}  // namespace ev

#endif  // !BASE_COLUMNFILE_H_
