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

typedef std::unique_ptr<DIR, int(*)(DIR*)> UniqueDIR;
typedef std::unique_ptr<FILE, decltype(&fclose)> UniqueFILE;

kj::AutoCloseFd OpenFile(const char* path, int flags, int mode = 0666);

kj::AutoCloseFd OpenFile(int dir_fd, const char* path, int flags,
                         int mode = 0666);

UniqueFILE OpenFileStream(const char* path, const char* mode);

void WriteAll(int fd, const StringRef& buffer);

template <typename T>
void WriteAll(int fd, const std::vector<T>& vec) {
  WriteAll(fd, ev::StringRef(reinterpret_cast<const char*>(vec.data()),
                             vec.size() * sizeof(T)));
}

// Reads an entire file into a buffer, preferrably by memory-mapping.
kj::Array<char> ReadFD(int fd);
kj::Array<char> ReadFile(const char* path);

// Writes an entire file from a buffer, using O_CREAT | O_TRUNC.
void WriteFile(const char* path, const StringRef& buffer, int mode = 0664);

}  // namespace ev

#endif  // !BASE_FILE_H_
