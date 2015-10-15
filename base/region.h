#ifndef BASE_REGION_H_
#define BASE_REGION_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <kj/debug.h>

#include "concurrency.h"
#include "macros.h"

// Region is a mechanism for allocating individual chunks of memory while
// releasing them only collectively. At any given time only one thread is
// allowed to allocate memory from a region.
//
// These restrictions make memory allocation from a region much faster than
// from a regular dynamic heap.
//
// To extend the utility of region allocation to a very important case of
// producer-consumer inter-thread communication it is allowed to create a
// region in one thread, allocate some objects with it, then pass a region
// reference to another thread and thus give up the right to allocate some
// more objects to that thread.
//
// In such a situation it is possible that already allocated objects from a
// given region are used by different threads. Such kind of shared usage is
// not considered to be the design objective of regions. However a thread
// might allocate some temporary objects in a region, hand it over to another
// thread and then just invoke destructors for the temporary objects. It has
// to be ensured that such a scenario will not result in a crash.
//
// So it is required to synchronize region release between threads. It would
// be nice to use std::shared_ptr<T> for this purpose. But shared_ptr's
// internally allocate shared state using another allocator which we would
// like to avoid in the first place by introducing regions. And also they do
// some internal synchronization that might be expensive.
//
// So regions use hand-crafted reference counting scheme to protect against
// concurrent release.

namespace ev {
namespace concurrency {

class RegionPool {
  class RegionImpl;

 public:
  class Region {
   public:
    Region() noexcept : region_impl_(nullptr) {}

    Region(RegionImpl* region_impl) noexcept : region_impl_(region_impl) {}

    Region(Region const& other) noexcept : region_impl_(other.region_impl_) {
      if (region_impl_) region_impl_->AddRef();
    }

    Region(Region&& other) noexcept : region_impl_(other.region_impl_) {
      other.region_impl_ = nullptr;
    }

    Region& operator=(Region const& other) noexcept {
      if (region_impl_) region_impl_->Release();
      region_impl_ = other.region_impl_;
      if (region_impl_) region_impl_->AddRef();
      return *this;
    }

    Region& operator=(Region&& other) noexcept {
      std::swap(region_impl_, other.region_impl_);
      return *this;
    }

    ~Region() noexcept {
      if (region_impl_) region_impl_->Release();
    }

    void* EphemeralAllocate(size_t size) {
      return region_impl_->Allocate(size);
    }

    void* Allocate(size_t size) {
      void* p = region_impl_->Allocate(size);
      region_impl_->AddRef();
      return p;
    }

    void Deallocate(void* p __attribute__((unused))) noexcept {
      region_impl_->Release();
    }

   private:
    friend bool operator==(const RegionPool::Region& lhs,
                           const RegionPool::Region& rhs) noexcept;
    friend bool operator!=(const RegionPool::Region& lhs,
                           const RegionPool::Region& rhs) noexcept;

    RegionImpl* region_impl_;
  };

  RegionPool(size_t pool_size, size_t region_pages)
      : pool_size_{pool_size},
        region_size_{region_pages * EV_PAGE_SIZE},
        data_size_{region_size_ - AlignSize(sizeof(RegionImpl))} {
    KJ_REQUIRE(pool_size > 0, "pool_size must be greater than 0");
    KJ_REQUIRE(region_pages > 0, "region_pages must be greater than 0");

    pool_.reset(new char[pool_size_ * region_size_]);

    free_list_ = nullptr;
    for (size_t i = 0; i < pool_size_; ++i) {
      char* ptr = &pool_[i * region_size_];
      RegionImpl* reg = new (ptr) RegionImpl(*this);
      reg->next_ = free_list_;
      free_list_ = reg;
    }
  }

  ~RegionPool() {
    for (size_t i = 0; i < pool_size_; ++i) {
      RegionImpl* reg = reinterpret_cast<RegionImpl*>(&pool_[i * region_size_]);
      if (reg->used_ || reg->overflow_)
        fprintf(
            stderr,
            "Oops, deleting RegionPool while some Region is still in use\n");
    }
  }

  Region GetRegion() { return Region(AllocRegion()); }

  void Stat() {
    uintmax_t regions = 0;
    uintmax_t use_count = 0;
    uintmax_t allocate_count = 0;
    uintmax_t overflow_count = 0;
    uintmax_t allocate_total = 0;

    for (size_t i = 0; i < pool_size_; ++i) {
      RegionImpl* reg = reinterpret_cast<RegionImpl*>(&pool_[i * region_size_]);
      if (reg->use_count_ == 0) continue;

      regions++;
      use_count += reg->use_count_;
      allocate_count += reg->allocate_count_;
      overflow_count += reg->overflow_count_;
      allocate_total += reg->allocate_total_;
    }

    fprintf(stderr,
            "Region Pool:\n"
            "  size - %zu, used - %ju, use count - %ju,\n"
            "  alloc count - %ju, avg per use - %ju, overflow count - %ju,\n"
            "  alloc total - %ju, avg per use - %ju, avg per alloc - %ju\n",
            pool_size_, regions, use_count, allocate_count,
            use_count ? allocate_count / use_count : 0, overflow_count,
            allocate_total, use_count ? allocate_total / use_count : 0,
            allocate_count ? allocate_total / allocate_count : 0);
  };

 private:
  using LockType = typename DefaultSynch::LockType;
  using CondVarType = typename DefaultSynch::CondVarType;

  static size_t AlignSize(size_t size) { return ev::AlignUp(size, 16); }

  class RegionImpl {
   public:
    RegionImpl(RegionPool& pool) noexcept : ref_count_{0},
                                            next_{nullptr},
                                            overflow_{nullptr},
                                            pool_{pool},
                                            used_{0} {}

    void AddRef() noexcept {
      ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void Release() noexcept {
      if (ref_count_.fetch_sub(1, std::memory_order_relaxed) == 1)
        pool_.ReleaseRegion(this);
    }

    void* Allocate(size_t size) {
      allocate_count_++;
      allocate_total_ += size;

      size = AlignSize(size);

      if ((pool_.data_size_ - used_) >= size) {
        char* p = data_ + used_;
        used_ += size;
        return p;
      }

      if (overflow_ && (overflow_->size_ - overflow_->used_) >= size) {
        char* p = overflow_->data_ + overflow_->used_;
        overflow_->used_ += size;
        return p;
      }

      return CreateOverflow(size);
    }

   private:
    friend class RegionPool;

    struct Overflow {
      Overflow* next_;
      size_t size_;
      size_t used_;
      char data_[];
    };

    void* CreateOverflow(size_t size) {
      overflow_count_++;

      static const size_t min_size = EV_PAGE_SIZE - 16 - sizeof(Overflow);
      size_t overflow_size = std::max(size, min_size);

      char* raw_overflow = new char[overflow_size + sizeof(Overflow)];
      Overflow* overflow = reinterpret_cast<Overflow*>(raw_overflow);

      overflow->next_ = overflow_;
      overflow->size_ = overflow_size;
      overflow->used_ = size;
      overflow_ = overflow;

      return overflow->data_;
    }

    void DestroyOverflow() noexcept {
      Overflow* overflow = overflow_;
      while (overflow != nullptr) {
        Overflow* next = overflow->next_;
        delete[] reinterpret_cast<char*>(overflow);
        overflow = next;
      }
    }

    std::atomic<size_t> ref_count_;
    RegionImpl* next_;
    Overflow* overflow_;
    RegionPool& pool_;
    size_t used_;

    uint64_t use_count_ = 0;
    uint64_t allocate_count_ = 0;
    uint64_t overflow_count_ = 0;
    uint64_t allocate_total_ = 0;

    char data_[];
  };

  std::atomic<RegionImpl*> free_list_;

  const size_t pool_size_;
  const size_t region_size_;
  const size_t data_size_;

  std::unique_ptr<char[]> pool_;

  LockType ready_lock_;
  CondVarType ready_cond_;

  // necessary to protect against ABA-problem in multi-producer case.
  RegionImpl* AllocRegion() {
    LockGuard<LockType> guard(ready_lock_);
    RegionImpl* list_head = free_list_.load(std::memory_order_relaxed);
    while (!list_head) {
      ready_cond_.Wait(guard);
      list_head = free_list_.load(std::memory_order_relaxed);
    }
    free_list_.store(list_head->next_, std::memory_order_relaxed);
    list_head->ref_count_.store(1, std::memory_order_relaxed);
    list_head->use_count_++;
    return list_head;
  }

  void ReleaseRegion(RegionImpl* region) {
    region->DestroyOverflow();
    region->overflow_ = nullptr;
    region->used_ = 0;

    LockGuard<LockType> guard(ready_lock_);
    region->next_ = free_list_.load(std::memory_order_relaxed);
    free_list_.store(region, std::memory_order_relaxed);
    if (region->next_ == nullptr) ready_cond_.NotifyOne();
  }
};

// This is a libstdc++ compatible allocator to let use region allocation
// with standard containers, strings, etc.
//
// Allocating and deallocating with this allocator maintains reference
// counting for the associated region. The region gets released only when
// all the references are released.
//
// The allocator itself also references the region. So all the allocated
// objects and the allocator as well should be destroyed to let the region
// go.
template <typename T>
class RegionAllocator {
 public:
  using value_type = T;
  using pointer = value_type*;

  template <typename U>
  struct rebind {
    using other = RegionAllocator<U>;
  };

  RegionAllocator() noexcept {}

  RegionAllocator(RegionPool::Region const& region) noexcept : region_(region) {
  }

  RegionAllocator(RegionPool::Region&& region) noexcept
      : region_(std::move(region)) {}

  template <typename U>
  RegionAllocator(RegionAllocator<U> const& alloc) noexcept
      : region_(alloc.GetRegion()) {}

  pointer allocate(size_t n) {
    return reinterpret_cast<pointer>(region_.Allocate(n * sizeof(value_type)));
  }

  void deallocate(pointer p, size_t n __attribute__((unused))) {
    region_.Deallocate(p);
  }

  RegionPool::Region const& GetRegion() const { return region_; }

 private:
  RegionPool::Region region_;
};

inline bool operator==(const RegionPool::Region& lhs,
                       const RegionPool::Region& rhs) noexcept {
  return lhs.region_impl_ == rhs.region_impl_;
}

inline bool operator!=(const RegionPool::Region& lhs,
                       const RegionPool::Region& rhs) noexcept {
  return lhs.region_impl_ != rhs.region_impl_;
}

template <class T>
inline bool operator==(const RegionAllocator<T>& lhs,
                       const RegionAllocator<T>& rhs) noexcept {
  return lhs.GetRegion() == rhs.GetRegion();
}

template <class T>
inline bool operator!=(const RegionAllocator<T>& lhs,
                       const RegionAllocator<T>& rhs) noexcept {
  return lhs.GetRegion() != rhs.GetRegion();
}

}  // namespace concurrency
}  // namespace ev

#endif  // BASE_REGION_H_
