//===-- sanitizer_quarantine.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Memory quarantine for AddressSanitizer and potentially other tools.
// Quarantine caches some specified amount of memory in per-thread caches,
// then evicts to global FIFO queue. When the queue reaches specified threshold,
// oldest memory is recycled.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_QUARANTINE_H
#define SANITIZER_QUARANTINE_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {

template<typename Node> class QuarantineCache;

// The callback interface is:
// Callback cb;
// Node *ptr;
// cb.Recycle(ptr);
template<typename Callback, typename Node>
class Quarantine {
 public:
  typedef QuarantineCache<Node> Cache;

  explicit Quarantine(LinkerInitialized)
      : cache_(LINKER_INITIALIZED) {
  }

  void Init(uptr size, uptr cache_size) {
    max_size_ = size;
    max_cache_size_ = cache_size;
  }

  void Put(Cache *c, Callback cb, Node *ptr) {
    c->Enqueue(ptr);
    if (c->Size() > max_cache_size_)
      Drain(c, cb);
  }

  void Drain(Cache *c, Callback cb) {
    SpinMutexLock l(&mutex_);
    while (Node *ptr = c->Dequeue())
      cache_.Enqueue(ptr);
    while (cache_.Size() > max_size_) {
      Node *ptr = cache_.Dequeue();
      cb.Recycle(ptr);
    }
  }

 private:
  SpinMutex mutex_;
  uptr max_size_;
  uptr max_cache_size_;
  Cache cache_;
};

// Per-thread cache of memory blocks (essentially FIFO queue).
template<typename Node>
class QuarantineCache {
 public:
  explicit QuarantineCache(LinkerInitialized) {
  }

  uptr Size() const {
    return size_;
  }

  void Enqueue(Node *ptr) {
    size_ += ptr->UsedSize();
    ptr->next = 0;
    if (tail_)
      tail_->next = ptr;
    else
      head_ = ptr;
    tail_ = ptr;
  }

  Node *Dequeue() {
    Node *ptr = head_;
    if (ptr == 0)
      return 0;
    head_ = ptr->next;
    if (head_ == 0)
      tail_ = 0;
    size_ -= ptr->UsedSize();
    return ptr;
  }

 private:
  Node *head_;
  Node *tail_;
  uptr size_;
};
}

#endif  // #ifndef SANITIZER_QUARANTINE_H
