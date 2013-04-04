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
// ASan-private header for asan_allocator2.cc.
//===----------------------------------------------------------------------===//

#ifndef ASAN_ALLOCATOR_H
#define ASAN_ALLOCATOR_H

#include "asan_internal.h"
#include "asan_interceptors.h"
#include "sanitizer_common/sanitizer_list.h"

namespace __asan {

enum AllocType {
  FROM_MALLOC = 1,  // Memory block came from malloc, calloc, realloc, etc.
  FROM_NEW = 2,     // Memory block came from operator new.
  FROM_NEW_BR = 3   // Memory block came from operator new [ ]
};

static const uptr kNumberOfSizeClasses = 255;
struct AsanChunk;

void InitializeAllocator();

class AsanChunkView {
 public:
  explicit AsanChunkView(AsanChunk *chunk) : chunk_(chunk) {}
  bool IsValid() { return chunk_ != 0; }
  uptr Beg();       // first byte of user memory.
  uptr End();       // last byte of user memory.
  uptr UsedSize();  // size requested by the user.
  uptr AllocTid();
  uptr FreeTid();
  void GetAllocStack(StackTrace *stack);
  void GetFreeStack(StackTrace *stack);
  bool AddrIsInside(uptr addr, uptr access_size, sptr *offset) {
    if (addr >= Beg() && (addr + access_size) <= End()) {
      *offset = addr - Beg();
      return true;
    }
    return false;
  }
  bool AddrIsAtLeft(uptr addr, uptr access_size, sptr *offset) {
    (void)access_size;
    if (addr < Beg()) {
      *offset = Beg() - addr;
      return true;
    }
    return false;
  }
  bool AddrIsAtRight(uptr addr, uptr access_size, sptr *offset) {
    if (addr + access_size > End()) {
      *offset = addr - End();
      return true;
    }
    return false;
  }

 private:
  AsanChunk *const chunk_;
};

AsanChunkView FindHeapChunkByAddress(uptr address);

// List of AsanChunks with total size.
class AsanChunkFifoList: public IntrusiveList<AsanChunk> {
 public:
  explicit AsanChunkFifoList(LinkerInitialized) { }
  AsanChunkFifoList() { clear(); }
  void Push(AsanChunk *n);
  void PushList(AsanChunkFifoList *q);
  AsanChunk *Pop();
  uptr size() { return size_; }
  void clear() {
    IntrusiveList<AsanChunk>::clear();
    size_ = 0;
  }
 private:
  uptr size_;
};

struct AsanThreadLocalMallocStorage {
  explicit AsanThreadLocalMallocStorage(LinkerInitialized x)
      { }
  AsanThreadLocalMallocStorage() {
    CHECK(REAL(memset));
    REAL(memset)(this, 0, sizeof(AsanThreadLocalMallocStorage));
  }

  uptr quarantine_cache[16];
  uptr allocator2_cache[96 * (512 * 8 + 16)];  // Opaque.
  void CommitBack();
};

// Fake stack frame contains local variables of one function.
// This struct should fit into a stack redzone (32 bytes).
struct FakeFrame {
  uptr magic;  // Modified by the instrumented code.
  uptr descr;  // Modified by the instrumented code.
  FakeFrame *next;
  u64 real_stack     : 48;
  u64 size_minus_one : 16;
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
  void Init(uptr stack_size);
  void StopUsingFakeStack() { alive_ = false; }
  void Cleanup();
  uptr AllocateStack(uptr size, uptr real_stack);
  static void OnFree(uptr ptr, uptr size, uptr real_stack);
  // Return the bottom of the maped region.
  uptr AddrIsInFakeStack(uptr addr);
  bool StackSize() { return stack_size_; }

 private:
  static const uptr kMinStackFrameSizeLog = 9;  // Min frame is 512B.
  static const uptr kMaxStackFrameSizeLog = 16;  // Max stack frame is 64K.
  static const uptr kMaxStackMallocSize = 1 << kMaxStackFrameSizeLog;
  static const uptr kNumberOfSizeClasses =
      kMaxStackFrameSizeLog - kMinStackFrameSizeLog + 1;

  bool AddrIsInSizeClass(uptr addr, uptr size_class);

  // Each size class should be large enough to hold all frames.
  uptr ClassMmapSize(uptr size_class);

  uptr ClassSize(uptr size_class) {
    return 1UL << (size_class + kMinStackFrameSizeLog);
  }

  void DeallocateFrame(FakeFrame *fake_frame);

  uptr ComputeSizeClass(uptr alloc_size);
  void AllocateOneSizeClass(uptr size_class);

  uptr stack_size_;
  bool   alive_;

  uptr allocated_size_classes_[kNumberOfSizeClasses];
  FakeFrameFifo size_classes_[kNumberOfSizeClasses];
  FakeFrameLifo call_stack_;
};

void *asan_memalign(uptr alignment, uptr size, StackTrace *stack,
                    AllocType alloc_type);
void asan_free(void *ptr, StackTrace *stack, AllocType alloc_type);

void *asan_malloc(uptr size, StackTrace *stack);
void *asan_calloc(uptr nmemb, uptr size, StackTrace *stack);
void *asan_realloc(void *p, uptr size, StackTrace *stack);
void *asan_valloc(uptr size, StackTrace *stack);
void *asan_pvalloc(uptr size, StackTrace *stack);

int asan_posix_memalign(void **memptr, uptr alignment, uptr size,
                          StackTrace *stack);
uptr asan_malloc_usable_size(void *ptr, StackTrace *stack);

uptr asan_mz_size(const void *ptr);
void asan_mz_force_lock();
void asan_mz_force_unlock();

void PrintInternalAllocatorStats();

}  // namespace __asan
#endif  // ASAN_ALLOCATOR_H
