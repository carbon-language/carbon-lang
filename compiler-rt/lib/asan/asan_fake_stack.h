//===-- asan_fake_stack.h ---------------------------------------*- C++ -*-===//
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
// ASan-private header for asan_fake_stack.cc
//===----------------------------------------------------------------------===//

#ifndef ASAN_FAKE_STACK_H
#define ASAN_FAKE_STACK_H

namespace __asan {

// Fake stack frame contains local variables of one function.
struct FakeFrame {
  uptr magic;  // Modified by the instrumented code.
  uptr descr;  // Modified by the instrumented code.
  uptr pc;     // Modified by the instrumented code.
  u64 real_stack     : 48;
  u64 class_id : 16;
  // End of the first 32 bytes.
  // The rest should not be used when the frame is active.
  FakeFrame *next;
};

struct FakeFrameFifo {
 public:
  void FifoPush(FakeFrame *node);
  FakeFrame *FifoPop();
 private:
  FakeFrame *first_, *last_;
};

template<uptr kMaxNumberOfFrames>
class FakeFrameLifo {
 public:
  explicit FakeFrameLifo(LinkerInitialized) {}
  FakeFrameLifo() : n_frames_(0) {}
  void LifoPush(FakeFrame *node) {
    CHECK_LT(n_frames_, kMaxNumberOfFrames);
    frames_[n_frames_++] = node;
  }
  void LifoPop() {
    CHECK(n_frames_);
    n_frames_--;
  }
  FakeFrame *top() {
    if (n_frames_ == 0)
      return 0;
    return frames_[n_frames_ - 1];
  }
 private:
  uptr n_frames_;
  FakeFrame *frames_[kMaxNumberOfFrames];
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
  void Init(uptr stack_size);
  void StopUsingFakeStack() { alive_ = false; }
  void Cleanup();
  uptr AllocateStack(uptr class_id, uptr size, uptr real_stack);
  static void OnFree(uptr ptr, uptr class_id, uptr size, uptr real_stack);
  // Return the bottom of the maped region.
  uptr AddrIsInFakeStack(uptr addr);
  uptr StackSize() const { return stack_size_; }

  static uptr ComputeSizeClass(uptr alloc_size);

  static uptr ClassSize(uptr class_id) {
    return 1UL << (class_id + kMinStackFrameSizeLog);
  }

 private:
  static const uptr kMinStackFrameSizeLog = 6;  // Min frame is 64B.
  static const uptr kMaxStackFrameSizeLog = 16;  // Max stack frame is 64K.
  static const uptr kNumberOfSizeClasses =
      kMaxStackFrameSizeLog - kMinStackFrameSizeLog + 1;
  // Must match the number of uses of DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID
  COMPILER_CHECK(kNumberOfSizeClasses == 11);
  static const uptr kMaxStackMallocSize = 1 << kMaxStackFrameSizeLog;
  static const uptr kMaxRecursionDepth = 60000;

  bool AddrIsInSizeClass(uptr addr, uptr class_id);

  // Each size class should be large enough to hold all frames.
  uptr ClassMmapSize(uptr class_id);

  void DeallocateFrame(FakeFrame *fake_frame);

  void AllocateOneSizeClass(uptr class_id);

  uptr stack_size_;
  bool   alive_;

  uptr allocated_size_classes_[kNumberOfSizeClasses];
  FakeFrameFifo size_classes_[kNumberOfSizeClasses];
  FakeFrameLifo<kMaxRecursionDepth> call_stack_;
};

COMPILER_CHECK(sizeof(FakeStack) <= (1 << 19));

}  // namespace __asan

#endif  // ASAN_FAKE_STACK_H
