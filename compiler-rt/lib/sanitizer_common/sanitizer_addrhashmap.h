//===-- sanitizer_addrhashmap.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Concurrent uptr->T hashmap.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ADDRHASHMAP_H
#define SANITIZER_ADDRHASHMAP_H

#include "sanitizer_common.h"
#include "sanitizer_mutex.h"
#include "sanitizer_atomic.h"

namespace __sanitizer {

// Concurrent uptr->T hashmap.
// T must be a POD type, kSize is preferrably a prime but can be any number.
// The hashmap is fixed size, it crashes on overflow.
// Usage example:
//
// typedef AddrHashMap<uptr, 11> Map;
// Map m;
// {
//   Map::Handle h(&m, addr);
//   use h.operator->() to access the data
//   if h.created() then the element was just created, and the current thread
//     has exclusive access to it
//   otherwise the current thread has only read access to the data
// }
// {
//   Map::Handle h(&m, addr, true);
//   this will remove the data from the map in Handle dtor
//   the current thread has exclusive access to the data
//   if !h.exists() then the element never existed
// }
template<typename T, uptr kSize>
class AddrHashMap {
 private:
  struct Cell {
    StaticSpinMutex  mtx;
    atomic_uintptr_t addr;
    T                val;
  };

 public:
  AddrHashMap();

  class Handle {
   public:
    Handle(AddrHashMap<T, kSize> *map, uptr addr, bool remove = false);
    ~Handle();
    T *operator -> ();
    bool created() const;
    bool exists() const;

   private:
    AddrHashMap<T, kSize> *map_;
    Cell                  *cell_;
    uptr                   addr_;
    bool                   created_;
    bool                   remove_;
  };

 private:
  friend class Handle;
  Cell *table_;

  static const uptr kLocked = 1;
  static const uptr kRemoved = 2;

  Cell *acquire(uptr addr, bool remove, bool *created);
  void  release(uptr addr, bool remove, bool created, Cell *c);
  uptr hash(uptr addr);
};

template<typename T, uptr kSize>
AddrHashMap<T, kSize>::Handle::Handle(AddrHashMap<T, kSize> *map, uptr addr,
    bool remove) {
  map_ = map;
  addr_ = addr;
  remove_ = remove;
  cell_ = map_->acquire(addr_, remove_, &created_);
}

template<typename T, uptr kSize>
AddrHashMap<T, kSize>::Handle::~Handle() {
  map_->release(addr_, remove_, created_, cell_);
}

template<typename T, uptr kSize>
T *AddrHashMap<T, kSize>::Handle::operator -> () {
  return &cell_->val;
}

template<typename T, uptr kSize>
bool AddrHashMap<T, kSize>::Handle::created() const {
  return created_;
}

template<typename T, uptr kSize>
bool AddrHashMap<T, kSize>::Handle::exists() const {
  return cell_ != 0;
}

template<typename T, uptr kSize>
AddrHashMap<T, kSize>::AddrHashMap() {
  table_ = (Cell*)MmapOrDie(kSize * sizeof(Cell), "AddrHashMap");
}

template<typename T, uptr kSize>
typename AddrHashMap<T, kSize>::Cell *AddrHashMap<T, kSize>::acquire(uptr addr,
    bool remove, bool *created) {
  // When we access the element associated with addr,
  // we lock its home cell (the cell associated with hash(addr).
  // If the element was just created or is going to be removed,
  // we lock the cell in write mode. Otherwise we lock in read mode.
  // The locking protects the object lifetime (it's not removed while
  // somebody else accesses it). And also it helps to resolve concurrent
  // inserts.
  // Note that the home cell is not necessary the cell where the element is
  // stored.
  *created = false;
  uptr h0 = hash(addr);
  Cell *c0 = &table_[h0];
  // First try to find an existing element w/o read mutex.
  {
    uptr h = h0;
    for (;;) {
      Cell *c = &table_[h];
      uptr addr1 = atomic_load(&c->addr, memory_order_acquire);
      if (addr1 == 0)  // empty cell denotes end of the cell chain for the elem
        break;
      // Locked cell means that another thread can be concurrently inserting
      // the same element, fallback to mutex.
      if (addr1 == kLocked)
        break;
      if (addr1 == addr)  // ok, found it
        return c;
      h++;
      if (h == kSize)
        h = 0;
      CHECK_NE(h, h0);  // made the full cycle
    }
  }
  if (remove)
    return 0;
  // Now try to create it under the mutex.
  c0->mtx.Lock();
  uptr h = h0;
  for (;;) {
    Cell *c = &table_[h];
    uptr addr1 = atomic_load(&c->addr, memory_order_acquire);
    if (addr1 == addr)  // another thread has inserted it ahead of us
      return c;
    // Skip kLocked, since we hold the home cell mutex, it can't be our elem.
    if ((addr1 == 0 || addr1 == kRemoved) &&
        atomic_compare_exchange_strong(&c->addr, &addr1, kLocked,
          memory_order_acq_rel)) {
      // we've created the element
      *created = true;
      return c;
    }
    h++;
    if (h == kSize)
      h = 0;
    CHECK_NE(h, h0);  // made the full cycle
  }
}

template<typename T, uptr kSize>
void AddrHashMap<T, kSize>::release(uptr addr, bool remove, bool created,
    Cell *c) {
  if (c == 0)
    return;
  // if we are going to remove, we must hold write lock
  uptr addr1 = atomic_load(&c->addr, memory_order_relaxed);
  if (created) {
    // denote completion of insertion
    atomic_store(&c->addr, addr, memory_order_release);
    // unlock the home cell
    uptr h0 = hash(addr);
    Cell *c0 = &table_[h0];
    c0->mtx.Unlock();
  } else {
    CHECK_EQ(addr, addr1);
    if (remove) {
      // denote that the cell is empty now
      atomic_store(&c->addr, kRemoved, memory_order_release);
    }
  }
}

template<typename T, uptr kSize>
uptr AddrHashMap<T, kSize>::hash(uptr addr) {
  addr += addr << 10;
  addr ^= addr >> 6;
  return addr % kSize;
}

} // namespace __sanitizer

#endif // SANITIZER_ADDRHASHMAP_H
