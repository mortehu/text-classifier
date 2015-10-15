#ifndef BASE_MACROS_H_
#define BASE_MACROS_H_ 1

#include <cstddef>
#include <cstdint>
#include <utility>

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

constexpr std::size_t operator"" _z(unsigned long long n) { return n; }

#if __GNUC__ >= 3
#define EV_UNLIKELY(cond) __builtin_expect((cond), 0)
#define EV_LIKELY(cond) __builtin_expect((cond), 1)
#define EV_USE_RESULT __attribute__((warn_unused_result))
#define EV_PACKED __attribute__((packed))
#else
#define EV_UNLIKELY(cond) (cond)
#define EV_LIKELY(cond) (cond)
#define EV_USE_RESULT
#define EV_PACKED
#endif

#define EV_PAGE_SIZE (4096)
#define EV_CACHELINE_SIZE (64)

namespace ev {

constexpr std::size_t pow(const std::size_t n, const std::size_t e) {
  return (e == 0) ? 1 : (e == 1) ? n : pow(n, e / 2) * pow(n, e - e / 2);
}

// Round down to a power of two multiple.
constexpr std::size_t Align(std::size_t n, std::size_t a) {
  return n & ~(a - 1);
}

// Round up to a power of two multiple.
constexpr std::size_t AlignUp(std::size_t n, std::size_t a) {
  return Align(n + a - 1, a);
}

// Return the first parameters whose boolean value is `true`.
template<typename T>
T Coalesce(T t) { return std::move(t); }

template<typename T, typename... Args>
T Coalesce(T t, Args... args) {
  return t ? std::move(t) : Coalesce(args...);
}

}  // namespace

#endif  // !BASE_MACROS_H_
