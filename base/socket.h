#warning "x"
#ifndef BASE_SOCKET_H_
#define BASE_SOCKET_H_ 1

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <sys/types.h>

#include <kj/io.h>

#include "base/file.h"

namespace ev {

// Returns a vector of sockets for listening on the given address and port.
// An vector is returned instead of a single object, because IPv4 and IPv6
// may require separate file descriptors.
std::vector<kj::AutoCloseFd> tcp_listen(const char* address,
                                        const char* service);

// Returns a reading socket and a writing socket, from the pipe system call.
std::pair<kj::AutoCloseFd, kj::AutoCloseFd> pipe(bool close_on_exec = true);

}  // namespace ev

#endif  // !BASE_SOCKET_H_
