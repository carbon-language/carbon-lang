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
  u64 size_minus_one : 16;
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
  FakeStack();
  explicit FakeStack(LinkerInitialized x) : call_stack_(x) {}
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
  static const uptr kMaxRecursionDepth = 1023;

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
  FakeFrameLifo<kMaxRecursionDepth> call_stack_;
};

}  // namespace __asan

#endif  // ASAN_FAKE_STACK_H
