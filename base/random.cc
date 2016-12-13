#include "base/random.h"

#include <chrono>
#include <memory>
#include <mutex>

namespace ev {

std::minstd_rand0 rng(
    std::chrono::system_clock::now().time_since_epoch().count());

}  // namespace ev
