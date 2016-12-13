#ifndef BASE_FILE_H_
#define BASE_FILE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <dirent.h>
#include <fcntl.h>

#include <kj/array.h>
#include <kj/debug.h>
#include <kj/io.h>

#include "base/stringref.h"

namespace ev {

typedef std::unique_ptr<DIR, decltype(&closedir)> UniqueDIR;
typedef std::unique_ptr<FILE, decltype(&fclose)> UniqueFILE;

kj::AutoCloseFd OpenFile(const char* path, int flags, int mode = 0666);

kj::AutoCloseFd OpenFile(int dir_fd, const char* path, int flags,
                         int mode = 0666);

UniqueDIR OpenDirectory(const char* path);

UniqueDIR OpenDirectory(int dir_fd, const char* path);

UniqueFILE OpenFileStream(const char* path, const char* mode);

std::pair<kj::AutoCloseFd, std::string> TemporaryFile(const char* base_name);

std::pair<kj::AutoCloseFd, std::string> TemporaryFile(int dir_fd,
                                                      const char* base_name);

// Creates a temporary directory with the provided prefix.
std::string TemporaryDirectory(const char* base_name);

// Creates a temporary directory in TMPDIR.
std::string TemporaryDirectory();

class DirectoryTreeRemover {
 public:
  // Creates an object that will remove the given directory tree on
  // destruction.  If `root' is an empty string, nothing is removed.
  DirectoryTreeRemover(std::string root);

  // Takes ownership of any directory tree owened by `rhs', and ensures `rhs'
  // does not own any directory tree.
  DirectoryTreeRemover(DirectoryTreeRemover&& rhs);

  // Recursively removes any currently associated directory tree.
  ~DirectoryTreeRemover();

  KJ_DISALLOW_COPY(DirectoryTreeRemover);

  // Recursively removes any currently associated directory tree, then takes
  // ownership over any directory tree owned by `rhs', and ensures `rhs' does
  // not own any directory tree.
  DirectoryTreeRemover& operator=(DirectoryTreeRemover&& rhs);

  // Returns the currently owned directory tree, or any empty string if none.
  const std::string& Root() const { return root_; }

  // Recursively removes any currently associated directory tree.
  void RemoveTree();

 private:
  void RemoveTree(const std::string& root);

  // The currently owned directory tree, or empty if none.
  std::string root_;
};

void FindFiles(const std::string& root, std::function<void(std::string&&)>&& callback);

// Reads from `fd' until end of file or error.
//
// Blocks of data are passed to the `callback' function.  If the `callback'
// function returns false, reading stops and the function returns false.  If an
// error occurs, an exception is thrown.  If the end of file is reached, the
// functions returns true.
bool ReadAll(int fd, std::function<bool(const void*, size_t)> callback);

void WriteAll(int fd, const StringRef& buffer);

template <typename T>
void WriteAll(int fd, const std::vector<T>& vec) {
  WriteAll(fd, ev::StringRef(reinterpret_cast<const char*>(vec.data()),
                             vec.size() * sizeof(T)));
}

size_t Read(int fd, void* dest, size_t size_min, size_t size_max);

size_t PRead(int fd, void* dest, size_t size_min, size_t size_max, off_t offset);

// Like the pread() system call, but throws an exception if not all the data
// can be read.
void PRead(int fd, void* dest, size_t size, off_t offset);

// Reads an entire file into a buffer, preferrably by memory-mapping.
kj::Array<char> ReadFD(int fd);
kj::Array<char> ReadFile(const char* path);

// Writes an entire file from a buffer, using O_CREAT | O_TRUNC.
void WriteFile(const char* path, const StringRef& buffer, int mode = 0664);

}  // namespace ev

#endif  // !BASE_FILE_H_
