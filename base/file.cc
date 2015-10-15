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

void MakeRandomString(char* output, size_t length) {
  static const char kLetters[] =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

  std::uniform_int_distribution<uint8_t> rng_dist(0, strlen(kLetters) - 1);

  while (length--) *output++ = kLetters[rng_dist(ev::StrongRNG())];
}

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

UniqueDIR OpenDirectory(const char* path) {
  DIR* result = opendir(path);
  if (!result) KJ_FAIL_SYSCALL("opendir", errno, path);
  return UniqueDIR(result, &closedir);
}

UniqueDIR OpenDirectory(int dir_fd, const char* path) {
  int fd;
  KJ_SYSCALL(fd = openat(dir_fd, path, O_RDONLY | O_DIRECTORY));
  DIR* result = fdopendir(fd);
  if (!result) {
    close(fd);
    KJ_FAIL_SYSCALL("fdopendir", errno, dir_fd, path);
  }
  return UniqueDIR(result, &closedir);
}

UniqueFILE OpenFileStream(const char* path, const char* mode) {
  return UniqueFILE(fopen(path, mode), fclose);
}

kj::AutoCloseFd AnonTemporaryFile(const char* path, int mode) {
  if (!path) {
    path = getenv("TMPDIR");
    if (!path) path = "/tmp";
  }

  return OpenFile(path, O_TMPFILE | O_RDWR, mode);
}

void LinkAnonTemporaryFile(int fd, const char* path) {
  LinkAnonTemporaryFile(AT_FDCWD, fd, path);
}

void LinkAnonTemporaryFile(int dir_fd, int fd, const char* path) {
  char temp_path[32];
  snprintf(temp_path, sizeof(temp_path), "/proc/self/fd/%d", fd);
  auto ret = linkat(AT_FDCWD, temp_path, dir_fd, path, AT_SYMLINK_FOLLOW);
  if (ret == 0) return;
  if (errno != EEXIST) {
    KJ_FAIL_SYSCALL("linkat", errno, temp_path, path);
  }

  // Target already exists, so we need an intermediate filename to atomically
  // replace with rename().
  std::string intermediate_path = path;
  intermediate_path += ".XXXXXX";

  static const size_t kMaxAttempts = 62 * 62 * 62;

  for (size_t i = 0; i < kMaxAttempts; ++i) {
    MakeRandomString(&intermediate_path[intermediate_path.size() - 6], 6);

    ret = linkat(AT_FDCWD, temp_path, dir_fd, intermediate_path.c_str(),
                 AT_SYMLINK_FOLLOW);

    if (ret == 0) {
      KJ_SYSCALL(renameat(dir_fd, intermediate_path.c_str(), dir_fd, path),
                 intermediate_path, path);
      return;
    }

    if (errno != EEXIST) {
      KJ_FAIL_SYSCALL("linkat", errno, intermediate_path, path);
    }
  }

  KJ_FAIL_REQUIRE("all temporary file creation attempts failed", kMaxAttempts);
}

std::pair<kj::AutoCloseFd, std::string> TemporaryFile(const char* base_name) {
  std::vector<char> path(base_name, base_name + strlen(base_name));
  path.push_back('.');
  for (size_t i = 0; i < 6; ++i) path.push_back('X');
  path.push_back(0);

  int fd;
  KJ_SYSCALL(fd = mkstemp(&path[0]), base_name);

  return std::make_pair(kj::AutoCloseFd(fd), &path[0]);
}

std::pair<kj::AutoCloseFd, std::string> TemporaryFile(int dir_fd,
                                                      const char* base_name) {
  std::string path(base_name);
  path += ".XXXXXX";

  static const size_t kMaxAttempts = 62 * 62 * 62;

  for (size_t i = 0; i < kMaxAttempts; ++i) {
    MakeRandomString(&path[path.size() - 6], 6);

    int fd = openat(dir_fd, path.c_str(), O_RDWR | O_CREAT | O_EXCL,
                    S_IRUSR | S_IWUSR);
    if (fd >= 0)
      return std::make_pair<kj::AutoCloseFd, std::string>(kj::AutoCloseFd(fd),
                                                          std::move(path));
    if (errno != EEXIST && errno != EPERM)
      KJ_FAIL_SYSCALL("open", errno, path, dir_fd, base_name);
  }

  KJ_FAIL_REQUIRE("all temporary file creation attempts failed", kMaxAttempts);
}

std::string TemporaryDirectory(const char* base_name) {
  char path[PATH_MAX];
  strcpy(path, base_name);
  strcat(path, ".XXXXXX");

  KJ_SYSCALL(mkdtemp(path));

  return path;
}

std::string TemporaryDirectory() {
  const char* tmpdir = getenv("TMPDIR");
  if (!tmpdir) tmpdir = "/tmp";
  return TemporaryDirectory(cat(tmpdir, "/ev-tmp").c_str());
}

DirectoryTreeRemover::DirectoryTreeRemover(std::string root)
    : root_(std::move(root)) {
  KJ_ASSERT(!root_.empty());
}

DirectoryTreeRemover::DirectoryTreeRemover(DirectoryTreeRemover&& rhs) {
  std::swap(root_, rhs.root_);
}

DirectoryTreeRemover::~DirectoryTreeRemover() {
  RemoveTree();
}

DirectoryTreeRemover& DirectoryTreeRemover::operator=(
    DirectoryTreeRemover&& rhs) {
  RemoveTree();
  std::swap(root_, rhs.root_);
  return *this;
}

void DirectoryTreeRemover::RemoveTree() {
  if (!root_.empty()) {
    RemoveTree(root_);
    root_.clear();
  }
}

void DirectoryTreeRemover::RemoveTree(const std::string& root) {
  auto dir = OpenDirectory(root.c_str());

  while (auto ent = readdir(dir.get())) {
    if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;

    std::string path = root;
    path.push_back('/');
    path.append(ent->d_name);

    if (ent->d_type == DT_UNKNOWN) {
      struct stat st;
      KJ_SYSCALL(stat(path.c_str(), &st));
      if (S_ISDIR(st.st_mode)) ent->d_type = DT_DIR;
    }

    if (ent->d_type == DT_DIR) {
      RemoveTree(path);
      continue;
    }

    KJ_SYSCALL(unlink(path.c_str()));
  }

  KJ_SYSCALL(rmdir(root.c_str()));
}

void FindFiles(const std::string& root, std::function<void(std::string&&)>&& callback) {
  auto dir = OpenDirectory(root.c_str());

  while (auto ent = readdir(dir.get())) {
    if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;

    std::string path = root;
    path.push_back('/');
    path.append(ent->d_name);

    if (ent->d_type == DT_UNKNOWN) {
      struct stat st;
      KJ_SYSCALL(stat(path.c_str(), &st));
      if (S_ISDIR(st.st_mode)) ent->d_type = DT_DIR;
    }

    if (ent->d_type == DT_DIR) {
      FindFiles(path, [&callback](auto&& path) { callback(std::move(path)); });
      continue;
    }

    callback(std::move(path));
  }
}

bool ReadAll(int fd, std::function<bool(const void*, size_t)> callback) {
  std::vector<char> buffer(65536);

  for (;;) {
    ssize_t ret;
    KJ_SYSCALL(ret = read(fd, &buffer[0], buffer.size()));
    if (!ret) return true;
    if (!callback(&buffer[0], ret)) return false;
  }
}

void WriteAll(int fd, const StringRef& buffer) {
  size_t offset = 0;

  while (offset < buffer.size()) {
    ssize_t ret;
    KJ_SYSCALL(ret = write(fd, buffer.data() + offset, buffer.size() - offset));
    offset += ret;
  }
}

size_t Read(int fd, void* dest, size_t size_min, size_t size_max) {
  size_t size = 0;
  while (size < size_max) {
    ssize_t ret;
    KJ_SYSCALL(ret = read(fd, dest, size_max - size));
    if (ret == 0) {
      KJ_REQUIRE(size >= size_min, size, size_min, size_max);
      return size;
    }
    size += ret;
    dest = reinterpret_cast<char*>(dest) + ret;
  }

  return size;
}

size_t PRead(int fd, void* dest, size_t size_min, size_t size_max, off_t offset) {
  size_t result = 0;

  while (size_max > 0) {
    ssize_t ret;
    KJ_SYSCALL(ret = pread(fd, dest, size_max, offset));
    if (ret == 0) break;
    result += ret;
    size_max -= ret;
    offset += ret;
    dest = reinterpret_cast<char*>(dest) + ret;
  }

  KJ_REQUIRE(result >= size_min, "unexpectedly reached end of file", offset, result, size_min);

  return result;
}

void PRead(int fd, void* dest, size_t size, off_t offset) { PRead(fd, dest, size, size, offset); }

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
