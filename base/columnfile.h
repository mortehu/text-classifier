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

class ColumnFileWriter {
 public:
  // Encodes an unsigned integer into a variable length integer.  Exposed here
  // for unit testing.
  static void PutInt(std::string& output, uint32_t value);

  ColumnFileWriter(kj::AutoCloseFd&& fd);

  ColumnFileWriter(const char* path, int mode = 0666);

  ColumnFileWriter(std::string& output);

  ~ColumnFileWriter();

  void SetCompression(ColumnFileCompression c) { compression_ = c; }

  // Inserts a value.
  //
  // The `level' indicates at which level in the schema hierarchy a new record
  // is inserted.
  void Put(uint8_t level, uint32_t column, const StringRef& data);
  void PutNull(uint8_t level, uint32_t column);

  void PutRow(const std::vector<std::pair<uint32_t, StringRef>>& row);

  // Writes all buffered records to the output stream.
  void Flush();

  // Finishes writing the file.  Returns the file descriptor in case the caller
  // wants it.
  //
  // This function is implicitly called by the destructor.
  kj::AutoCloseFd Finalize();

 private:
  class FieldWriter {
   public:
    void Put(uint8_t level, const StringRef& data);

    void PutNull(uint8_t level);

    void Flush();

    void Finalize(ColumnFileCompression compression);

    StringRef Data() const { return data_; }

   private:
    std::string data_;

    std::string value_;
    bool value_is_null_ = false;
    uint8_t level_ = 0;

    uint32_t repeat_ = 0;

    unsigned int shared_prefix_ = 0;
  };

  kj::AutoCloseFd fd_;
  std::string* output_string_ = nullptr;

  ColumnFileCompression compression_ = kColumnFileCompressionSnappy;

  std::map<uint32_t, FieldWriter> fields_;
};

class ColumnFileReader {
 public:
  // Decodes an integer encoded with ColumnFileWriter::PutInt().
  static uint32_t GetInt(StringRef& input);

  // Reads a column file as a stream.  If you want to use memory-mapped I/O,
  // use the StringRef based constructor below.
  ColumnFileReader(kj::AutoCloseFd&& fd);

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

  const std::vector<std::pair<uint32_t, StringRef>>& GetRow();

  void SeekToStart();

 private:
  class FieldReader {
   public:
    FieldReader(std::unique_ptr<char[]> buffer, size_t buffer_size,
                ColumnFileCompression compression);

    FieldReader(StringRef data, ColumnFileCompression compression);

    bool End() const { return !repeat_ && data_.empty(); }

    uint8_t NextLevel() {
      if (!repeat_) {
        KJ_ASSERT(!data_.empty());
        Fill();
      }
      return level_;
    }

    const StringRef* Get() {
      if (!repeat_) {
        KJ_ASSERT(!data_.empty());
        Fill();
      }
      --repeat_;

      return value_is_null_ ? nullptr : &value_;
    }

   private:
    void Fill();

    std::unique_ptr<char[]> buffer_;
    size_t buffer_size_ = 0;

    StringRef data_;

    ColumnFileCompression compression_;

    StringRef value_;
    bool value_is_null_ = true;
    uint8_t level_;

    uint32_t repeat_ = 0;
  };

  void Fill();

  std::unordered_set<uint32_t> column_filter_;

  std::string buffer_;

  StringRef data_;

  std::vector<std::pair<uint32_t, FieldReader>> fields_;

  kj::AutoCloseFd fd_;

  std::vector<std::pair<uint32_t, StringRef>> row_buffer_;
};

}  // namespace ev

#endif  // !BASE_COLUMNFILE_H_
