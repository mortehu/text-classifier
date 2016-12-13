#ifndef BASE_RANDOM_H_
#define BASE_RANDOM_H_

#include <random>

namespace ev {

// Shared fast PRNG initialized with weak entropy at startup.
extern std::minstd_rand0 rng;

}  // namespace ev

#endif
