#include "base/random.h"

#include <chrono>
#include <memory>
#include <mutex>

namespace ev {

std::minstd_rand0 rng(
    std::chrono::system_clock::now().time_since_epoch().count());

std::mt19937_64& StrongRNG() {
  static std::mutex mutex;
  static std::unique_ptr<std::mt19937_64> rng;
  if (!rng) {
    std::unique_lock<std::mutex> lk(mutex);
    if (!rng) {
      std::random_device rdev;
      rng = std::make_unique<std::mt19937_64>(rdev());
    }
  }

  return *rng;
}

}  // namespace ev
