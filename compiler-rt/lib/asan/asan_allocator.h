//===-- asan_allocator.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header for asan_allocator.cc.
//===----------------------------------------------------------------------===//

#ifndef ASAN_ALLOCATOR_H
#define ASAN_ALLOCATOR_H

#include "asan_internal.h"
#include "asan_interceptors.h"

namespace __asan {

static const size_t kNumberOfSizeClasses = 255;
struct AsanChunk;

class AsanChunkFifoList {
 public:
  explicit AsanChunkFifoList(LinkerInitialized) { }
  AsanChunkFifoList() { clear(); }
  void Push(AsanChunk *n);
  void PushList(AsanChunkFifoList *q);
  AsanChunk *Pop();
  size_t size() { return size_; }
  void clear() {
    first_ = last_ = NULL;
    size_ = 0;
  }
 private:
  AsanChunk *first_;
  AsanChunk *last_;
  size_t size_;
};

struct AsanThreadLocalMallocStorage {
  explicit AsanThreadLocalMallocStorage(LinkerInitialized x)
      : quarantine_(x) { }
  AsanThreadLocalMallocStorage() {
    CHECK(REAL(memset));
    REAL(memset)(this, 0, sizeof(AsanThreadLocalMallocStorage));
  }

  AsanChunkFifoList quarantine_;
  AsanChunk *free_lists_[kNumberOfSizeClasses];
  void CommitBack();
};

// Fake stack frame contains local variables of one function.
// This struct should fit into a stack redzone (32 bytes).
struct FakeFrame {
  uintptr_t magic;  // Modified by the instrumented code.
  uintptr_t descr;  // Modified by the instrumented code.
  FakeFrame *next;
  uint64_t real_stack     : 48;
  uint64_t size_minus_one : 16;
};

struct FakeFrameFifo {
 public:
  void FifoPush(FakeFrame *node);
  FakeFrame *FifoPop();
 private:
  FakeFrame *first_, *last_;
};

class FakeFrameLifo {
 public:
  void LifoPush(FakeFrame *node) {
    node->next = top_;
    top_ = node;
  }
  void LifoPop() {
    CHECK(top_);
    top_ = top_->next;
  }
  FakeFrame *top() { return top_; }
 private:
  FakeFrame *top_;
};

// For each thread we create a fake stack and place stack objects on this fake
// stack instead of the real stack. The fake stack is not really a stack but
// a fast malloc-like allocator so that when a function exits the fake stack
// is not poped but remains there for quite some time until gets used again.
// So, we poison the objects on the fake stack when function returns.
// It helps us find use-after-return bugs.
// We can not rely on __asan_stack_free being called on every function exit,
// so we maintain a lifo list of all current fake frames and update it on every
// call to __asan_stack_malloc.
class FakeStack {
 public:
  FakeStack();
  explicit FakeStack(LinkerInitialized) {}
  void Init(size_t stack_size);
  void StopUsingFakeStack() { alive_ = false; }
  void Cleanup();
  uintptr_t AllocateStack(size_t size, size_t real_stack);
  static void OnFree(size_t ptr, size_t size, size_t real_stack);
  // Return the bottom of the maped region.
  uintptr_t AddrIsInFakeStack(uintptr_t addr);
  bool StackSize() { return stack_size_; }
 private:
  static const size_t kMinStackFrameSizeLog = 9;  // Min frame is 512B.
  static const size_t kMaxStackFrameSizeLog = 16;  // Max stack frame is 64K.
  static const size_t kMaxStackMallocSize = 1 << kMaxStackFrameSizeLog;
  static const size_t kNumberOfSizeClasses =
      kMaxStackFrameSizeLog - kMinStackFrameSizeLog + 1;

  bool AddrIsInSizeClass(uintptr_t addr, size_t size_class);

  // Each size class should be large enough to hold all frames.
  size_t ClassMmapSize(size_t size_class);

  size_t ClassSize(size_t size_class) {
    return 1UL << (size_class + kMinStackFrameSizeLog);
  }

  void DeallocateFrame(FakeFrame *fake_frame);

  size_t ComputeSizeClass(size_t alloc_size);
  void AllocateOneSizeClass(size_t size_class);

  size_t stack_size_;
  bool   alive_;

  uintptr_t allocated_size_classes_[kNumberOfSizeClasses];
  FakeFrameFifo size_classes_[kNumberOfSizeClasses];
  FakeFrameLifo call_stack_;
};

void *asan_memalign(size_t alignment, size_t size, AsanStackTrace *stack);
void asan_free(void *ptr, AsanStackTrace *stack);

void *asan_malloc(size_t size, AsanStackTrace *stack);
void *asan_calloc(size_t nmemb, size_t size, AsanStackTrace *stack);
void *asan_realloc(void *p, size_t size, AsanStackTrace *stack);
void *asan_valloc(size_t size, AsanStackTrace *stack);
void *asan_pvalloc(size_t size, AsanStackTrace *stack);

int asan_posix_memalign(void **memptr, size_t alignment, size_t size,
                          AsanStackTrace *stack);
size_t asan_malloc_usable_size(void *ptr, AsanStackTrace *stack);

size_t asan_mz_size(const void *ptr);
void asan_mz_force_lock();
void asan_mz_force_unlock();
void DescribeHeapAddress(uintptr_t addr, size_t access_size);

}  // namespace __asan
#endif  // ASAN_ALLOCATOR_H
