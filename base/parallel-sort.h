#ifndef BASE_PARALLEL_SORT_H_
#define BASE_PARALLEL_SORT_H_

#include <future>
#include <deque>
#include <thread>

#include "base/thread-pool.h"

namespace ev {

template <typename Iterator, typename Comparator>
void ParallelSort(Iterator begin, Iterator end, Comparator comparator,
                  ev::ThreadPool& thread_pool) {
  size_t count = std::distance(begin, end);

  if (count < 16) {
    std::sort(begin, end, comparator);
    return;
  }

  auto concurrency = thread_pool.Size();

  std::deque<std::future<std::pair<size_t, size_t>>> ranges;

  for (size_t i = 0; i < concurrency; ++i) {
    size_t range_begin = i * count / concurrency;
    size_t range_end = (i + 1) * count / concurrency;

    ranges.emplace_back(
        thread_pool.Launch([begin, range_begin, range_end, &comparator] {
          std::sort(begin + range_begin, begin + range_end, comparator);
          return std::make_pair(range_begin, range_end);
        }));
  }

  // TODO(mortehu): If more speed is desired, switch to a scheme that doesn't
  // do a final (or initial) pass over the entire array on a single core.

  while (ranges.size() > 1) {
    std::deque<std::future<std::pair<size_t, size_t>>> merged_ranges;

    while (!ranges.empty()) {
      auto range_a_promise = std::move(ranges.front());
      ranges.pop_front();

      if (ranges.empty()) {
        merged_ranges.emplace_back(std::move(range_a_promise));
        break;
      }

      auto range_b_promise = std::move(ranges.front());
      ranges.pop_front();

      merged_ranges.emplace_back(thread_pool.Launch([
        begin,
        range_a_promise = std::move(range_a_promise),
        range_b_promise = std::move(range_b_promise),
        &comparator
      ]() mutable {
        auto range_a = range_a_promise.get();
        auto range_b = range_b_promise.get();
        std::inplace_merge(begin + range_a.first, begin + range_a.second,
                           begin + range_b.second, comparator);
        return std::make_pair(range_a.first, range_b.second);
      }));
    }

    ranges.swap(merged_ranges);
  }

  auto final_range = ranges[0].get();
  KJ_REQUIRE(final_range.first == 0);
  KJ_REQUIRE(final_range.second == count);
}

}  // namespace ev

#endif  // !BASE_PARALLEL_SORT_H_
