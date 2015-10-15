#ifndef BASE_RANDOM_H_
#define BASE_RANDOM_H_

#include <random>

namespace ev {

// Shared fast PRNG initialized with weak entropy at startup.
extern std::minstd_rand0 rng;

// Shared strong PRNG, seeded using the hardware RNG when it's first called.
std::mt19937_64& StrongRNG();

}  // namespace ev

#endif
