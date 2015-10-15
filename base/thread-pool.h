#ifndef BASE_THREAD_POOL_H_
#define BASE_THREAD_POOL_H_ 1

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "base/delegate.h"

namespace ev {

// Creates a fixed number of threads, used for asynchronous task execution.
// Example use:
//
//   ev::ThreadPool pool;
//
//   auto a = pool.Launch([] { return Job(0); });
//   auto b = pool.Launch([] { return Job(1); });
//
//   printf("Result: %d %d\n", a.get(), b.get());
//
// This is basically a replacement for std::async(std::launch::async, ...),
// which creates a thread on every invocation, which doesn't scale well.
class ThreadPool {
 public:
  // Constructs a thread pool with the given number of threads.
  ThreadPool(size_t n, size_t max_backlog = 256)
      : max_backlog_(max_backlog),
        caller_exec_count_(0),
        thread_exec_count_(n, 0) {
    while (n-- > 0)
      threads_.emplace_back(
          std::thread(std::bind(&ThreadPool::ThreadMain, this, n)));
  }

  // Constructs a thread pool with the same number of threads as supported by
  // the hardware.
  ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}

  // Instructs all worker threads to stop, waits for them to stop, then
  // destroys the thread pool.  Call Wait() before destruction if you want to
  // ensure all tasks have completed.
  ~ThreadPool() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_ = true;
    queue_not_empty_cv_.notify_all();
    lock.unlock();

    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
  }

  // Schedules a void task for asynchronous execution.
  template <class Function,
            typename std::enable_if<
                std::is_void<typename std::result_of<Function()>::type>::value,
                void>::type* = nullptr>
  void Launch(Function&& f) {
    std::unique_lock<std::mutex> lock(mutex_);

    // If we've reached the backlog limit, we just execute in the context of
    // the calling thread.
    if (queued_calls_.size() >= max_backlog_) {
      caller_exec_count_++;
      lock.unlock();
      f();
      return;
    }

    queued_calls_.emplace_back(std::move(f));

    ++scheduled_tasks_;
    lock.unlock();

    queue_not_empty_cv_.notify_one();
  }

  // Schedules a task for asynchronous execution.
  template <class Function,
            typename std::enable_if<
                !std::is_void<typename std::result_of<Function()>::type>::value,
                void>::type* = nullptr>
  std::future<typename std::result_of<Function()>::type> Launch(Function&& f) {
    std::promise<typename std::result_of<Function()>::type> promise;
    auto result = promise.get_future();

    std::unique_lock<std::mutex> lock(mutex_);

    // If we've reached the backlog limit, we just execute in the context of
    // the calling thread.
    if (queued_calls_.size() >= max_backlog_) {
      caller_exec_count_++;
      lock.unlock();
      promise.set_value(f());
      return result;
    }

    queued_calls_.emplace_back(
        [ f = std::move(f), promise = std::move(promise) ]() mutable {
          promise.set_value(f());
        });

    ++scheduled_tasks_;
    lock.unlock();

    queue_not_empty_cv_.notify_one();

    return result;
  }

  // Returns the number of threads in this thread pool.
  size_t Size() const { return threads_.size(); }

  // Waits for completion of all scheduled tasks.
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    completion_cv_.wait(
        lock, [this] { return completed_tasks_ == scheduled_tasks_; });
  }

  void Stat() {
    for (size_t i = 0; i < thread_exec_count_.size(); i++)
      printf("thread %lu: %lu\n", (unsigned long)i,
             (unsigned long)thread_exec_count_[i]);
    printf("caller: %lu\n", (unsigned long)caller_exec_count_);
  }

 private:
  // Worker thread entry point.  Runs tasks added to the queue until `done_' is
  // set to true by the destructor.
  void ThreadMain(size_t index) {
    std::unique_lock<std::mutex> lock(mutex_);

    for (;;) {
      queue_not_empty_cv_.wait(
          lock, [this] { return done_ || !queued_calls_.empty(); });
      if (done_) break;

      Delegate<void()> call(std::move(queued_calls_.front()));
      queued_calls_.pop_front();

      lock.unlock();

      call();

      thread_exec_count_[index]++;

      lock.lock();

      if (++completed_tasks_ == scheduled_tasks_) completion_cv_.notify_all();
    }
  }

  // The number of tasks added with Launch().
  size_t scheduled_tasks_ = 0;

  // The number of completed tasks.
  size_t completed_tasks_ = 0;

  // The maximum number of queued tasks, before we start running tasks in the
  // calling thread.
  size_t max_backlog_;

  std::vector<std::thread> threads_;

  uint64_t caller_exec_count_;
  std::vector<uint64_t> thread_exec_count_;

  std::condition_variable queue_not_empty_cv_;
  std::condition_variable completion_cv_;
  std::mutex mutex_;

  bool done_ = false;

  std::deque<Delegate<void()>> queued_calls_;
};

}  // namespace ev

#endif  // !BASE_THREAD_POOL_H_
