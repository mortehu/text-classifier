#ifndef BASE_ERROR_H_
#define BASE_ERROR_H_ 1

#include <kj/debug.h>

#define EV_FAIL_SYSCALL(syscall, object, ...)                    \
  do {                                                           \
    switch (errno) {                                             \
      case ENOENT:                                               \
        throw FileNotFoundException(object, __FILE__, __LINE__); \
      default:                                                   \
        KJ_FAIL_SYSCALL(syscall, errno, object, ##__VA_ARGS__);  \
    }                                                            \
  } while (0)

namespace ev {

class FileNotFoundException : public kj::Exception {
 public:
  FileNotFoundException(const char* path, const char* file, int line);
};

}  // namespace ev

#endif  // !BASE_ERROR_H_
