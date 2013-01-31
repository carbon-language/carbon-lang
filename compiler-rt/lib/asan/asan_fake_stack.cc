//===-- asan_fake_stack.cc ------------------------------------------------===//
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
// FakeStack is used to detect use-after-return bugs.
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"

namespace __asan {

FakeStack::FakeStack() {
  CHECK(REAL(memset) != 0);
  REAL(memset)(this, 0, sizeof(*this));
}

bool FakeStack::AddrIsInSizeClass(uptr addr, uptr size_class) {
  uptr mem = allocated_size_classes_[size_class];
  uptr size = ClassMmapSize(size_class);
  bool res = mem && addr >= mem && addr < mem + size;
  return res;
}

uptr FakeStack::AddrIsInFakeStack(uptr addr) {
  for (uptr i = 0; i < kNumberOfSizeClasses; i++) {
    if (AddrIsInSizeClass(addr, i)) return allocated_size_classes_[i];
  }
  return 0;
}

// We may want to compute this during compilation.
inline uptr FakeStack::ComputeSizeClass(uptr alloc_size) {
  uptr rounded_size = RoundUpToPowerOfTwo(alloc_size);
  uptr log = Log2(rounded_size);
  CHECK(alloc_size <= (1UL << log));
  if (!(alloc_size > (1UL << (log-1)))) {
    Printf("alloc_size %zu log %zu\n", alloc_size, log);
  }
  CHECK(alloc_size > (1UL << (log-1)));
  uptr res = log < kMinStackFrameSizeLog ? 0 : log - kMinStackFrameSizeLog;
  CHECK(res < kNumberOfSizeClasses);
  CHECK(ClassSize(res) >= rounded_size);
  return res;
}

void FakeFrameFifo::FifoPush(FakeFrame *node) {
  CHECK(node);
  node->next = 0;
  if (first_ == 0 && last_ == 0) {
    first_ = last_ = node;
  } else {
    CHECK(first_);
    CHECK(last_);
    last_->next = node;
    last_ = node;
  }
}

FakeFrame *FakeFrameFifo::FifoPop() {
  CHECK(first_ && last_ && "Exhausted fake stack");
  FakeFrame *res = 0;
  if (first_ == last_) {
    res = first_;
    first_ = last_ = 0;
  } else {
    res = first_;
    first_ = first_->next;
  }
  return res;
}

void FakeStack::Init(uptr stack_size) {
  stack_size_ = stack_size;
  alive_ = true;
}

void FakeStack::Cleanup() {
  alive_ = false;
  for (uptr i = 0; i < kNumberOfSizeClasses; i++) {
    uptr mem = allocated_size_classes_[i];
    if (mem) {
      PoisonShadow(mem, ClassMmapSize(i), 0);
      allocated_size_classes_[i] = 0;
      UnmapOrDie((void*)mem, ClassMmapSize(i));
    }
  }
}

uptr FakeStack::ClassMmapSize(uptr size_class) {
  return RoundUpToPowerOfTwo(stack_size_);
}

void FakeStack::AllocateOneSizeClass(uptr size_class) {
  CHECK(ClassMmapSize(size_class) >= GetPageSizeCached());
  uptr new_mem = (uptr)MmapOrDie(
      ClassMmapSize(size_class), __FUNCTION__);
  // Printf("T%d new_mem[%zu]: %p-%p mmap %zu\n",
  //       asanThreadRegistry().GetCurrent()->tid(),
  //       size_class, new_mem, new_mem + ClassMmapSize(size_class),
  //       ClassMmapSize(size_class));
  uptr i;
  for (i = 0; i < ClassMmapSize(size_class);
       i += ClassSize(size_class)) {
    size_classes_[size_class].FifoPush((FakeFrame*)(new_mem + i));
  }
  CHECK(i == ClassMmapSize(size_class));
  allocated_size_classes_[size_class] = new_mem;
}

uptr FakeStack::AllocateStack(uptr size, uptr real_stack) {
  if (!alive_) return real_stack;
  CHECK(size <= kMaxStackMallocSize && size > 1);
  uptr size_class = ComputeSizeClass(size);
  if (!allocated_size_classes_[size_class]) {
    AllocateOneSizeClass(size_class);
  }
  FakeFrame *fake_frame = size_classes_[size_class].FifoPop();
  CHECK(fake_frame);
  fake_frame->size_minus_one = size - 1;
  fake_frame->real_stack = real_stack;
  while (FakeFrame *top = call_stack_.top()) {
    if (top->real_stack > real_stack) break;
    call_stack_.LifoPop();
    DeallocateFrame(top);
  }
  call_stack_.LifoPush(fake_frame);
  uptr ptr = (uptr)fake_frame;
  PoisonShadow(ptr, size, 0);
  return ptr;
}

void FakeStack::DeallocateFrame(FakeFrame *fake_frame) {
  CHECK(alive_);
  uptr size = fake_frame->size_minus_one + 1;
  uptr size_class = ComputeSizeClass(size);
  CHECK(allocated_size_classes_[size_class]);
  uptr ptr = (uptr)fake_frame;
  CHECK(AddrIsInSizeClass(ptr, size_class));
  CHECK(AddrIsInSizeClass(ptr + size - 1, size_class));
  size_classes_[size_class].FifoPush(fake_frame);
}

void FakeStack::OnFree(uptr ptr, uptr size, uptr real_stack) {
  FakeFrame *fake_frame = (FakeFrame*)ptr;
  CHECK(fake_frame->magic = kRetiredStackFrameMagic);
  CHECK(fake_frame->descr != 0);
  CHECK(fake_frame->size_minus_one == size - 1);
  PoisonShadow(ptr, size, kAsanStackAfterReturnMagic);
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

uptr __asan_stack_malloc(uptr size, uptr real_stack) {
  if (!flags()->use_fake_stack) return real_stack;
  AsanThread *t = asanThreadRegistry().GetCurrent();
  if (!t) {
    // TSD is gone, use the real stack.
    return real_stack;
  }
  uptr ptr = t->fake_stack().AllocateStack(size, real_stack);
  // Printf("__asan_stack_malloc %p %zu %p\n", ptr, size, real_stack);
  return ptr;
}

void __asan_stack_free(uptr ptr, uptr size, uptr real_stack) {
  if (!flags()->use_fake_stack) return;
  if (ptr != real_stack) {
    FakeStack::OnFree(ptr, size, real_stack);
  }
}
