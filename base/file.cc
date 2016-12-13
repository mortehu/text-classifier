#include "base/file.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <err.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <kj/debug.h>

#include "base/cat.h"
#include "base/error.h"
#include "base/random.h"

namespace ev {

namespace {

// Disposes an array by calling the C library function "free".
class FreeArrayDisposer : public kj::ArrayDisposer {
 public:
  static const FreeArrayDisposer instance;

  FreeArrayDisposer() {}

  void disposeImpl(void* firstElement, size_t elementSize, size_t elementCount,
                   size_t capacity,
                   void (*destroyElement)(void*)) const override {
    KJ_REQUIRE(destroyElement == nullptr, destroyElement);
    ::free(firstElement);
  }
};

// Disposes an array by calling munmap.
class MunmapArrayDisposer : public kj::ArrayDisposer {
 public:
  static const MunmapArrayDisposer instance;

  MunmapArrayDisposer() {}

  void disposeImpl(void* firstElement, size_t elementSize, size_t elementCount,
                   size_t capacity,
                   void (*destroyElement)(void*)) const override {
    KJ_REQUIRE(destroyElement == nullptr, destroyElement);
    KJ_SYSCALL(munmap(firstElement, elementSize * elementCount));
  }
};

const FreeArrayDisposer FreeArrayDisposer::instance;
const MunmapArrayDisposer MunmapArrayDisposer::instance;

}  // namespace

kj::AutoCloseFd OpenFile(const char* path, int flags, int mode) {
  auto fd = open(path, flags, mode);
  if (fd == -1) EV_FAIL_SYSCALL("open", path, flags, mode);
  return kj::AutoCloseFd(fd);
}

kj::AutoCloseFd OpenFile(int dir_fd, const char* path, int flags, int mode) {
  auto fd = openat(dir_fd, path, flags, mode);
  if (fd == -1) EV_FAIL_SYSCALL("openat", path, dir_fd, flags, mode);
  return kj::AutoCloseFd(fd);
}

UniqueFILE OpenFileStream(const char* path, const char* mode) {
  return UniqueFILE(fopen(path, mode), fclose);
}

void WriteAll(int fd, const StringRef& buffer) {
  size_t offset = 0;

  while (offset < buffer.size()) {
    ssize_t ret;
    KJ_SYSCALL(ret = write(fd, buffer.data() + offset, buffer.size() - offset));
    offset += ret;
  }
}

kj::Array<char> ReadFD(int fd) {
  static const size_t kMinBufferSize = 1024 * 1024;

  off_t size = lseek(fd, 0, SEEK_END);
  if (size == -1) {
    if (errno != ESPIPE) {
      KJ_SYSCALL("lseek", errno);
    }

    // Unseekable file descriptor; let's just read into a vector.

    std::unique_ptr<char[], decltype(& ::free)> buffer(nullptr, ::free);
    size_t size = 0, alloc = 0;
    for (;;) {
      if (size == alloc) {
        alloc = size + kMinBufferSize;
        auto old_buffer = buffer.release();
        std::unique_ptr<char[], decltype(& ::free)> new_buffer(
            reinterpret_cast<char*>(realloc(old_buffer, alloc)), ::free);
        if (!new_buffer) {
          free(old_buffer);
          KJ_FAIL_SYSCALL("realloc", errno, alloc);
        }
        buffer = std::move(new_buffer);
      }

      size_t read_amount = alloc - size;
      ssize_t ret = ::read(fd, &buffer.get()[size], read_amount);
      if (ret < 0) KJ_FAIL_SYSCALL("read", errno);
      if (ret == 0) break;
      size += ret;
    }

    return kj::Array<char>(buffer.release(), size, FreeArrayDisposer::instance);
  } else if (size == 0) {
    return kj::Array<char>();
  } else {
    void* map = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) KJ_FAIL_SYSCALL("mmap", errno);

    return kj::Array<char>(reinterpret_cast<char*>(map), size,
                           MunmapArrayDisposer::instance);
  }
}

kj::Array<char> ReadFile(const char* path) {
  return ReadFD(OpenFile(path, O_RDONLY).get());
}

void WriteFile(const char* path, const StringRef& buffer, int mode) {
  WriteAll(OpenFile(path, O_WRONLY | O_CREAT | O_TRUNC, mode).get(), buffer);
}

}  // namespace ev
