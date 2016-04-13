#ifndef BASE_COMPRESSION_H_
#define BASE_COMPRESSION_H_ 1

#include <string>

namespace ev {

class StringRef;
class ThreadPool;

void CompressZLIB(std::string& output, const ev::StringRef& input, ev::ThreadPool& thread_pool);

}  // namespace ev

#endif  // !BASE_COMPRESSION_H_
