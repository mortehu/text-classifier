#include "base/error.h"

namespace ev {

FileNotFoundException::FileNotFoundException(const char* path, const char* file,
                                             int line)
    : kj::Exception(kj::Exception::Type::FAILED, file, line,
                    kj::heapString("File not found")) {}

}  // namespace ev
