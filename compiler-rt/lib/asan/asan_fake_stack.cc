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
#include "asan_poisoning.h"
#include "asan_thread.h"

namespace __asan {

bool FakeStack::AddrIsInSizeClass(uptr addr, uptr class_id) {
  uptr mem = allocated_size_classes_[class_id];
  uptr size = ClassMmapSize(class_id);
  bool res = mem && addr >= mem && addr < mem + size;
  return res;
}

uptr FakeStack::AddrIsInFakeStack(uptr addr) {
  for (uptr class_id = 0; class_id < kNumberOfSizeClasses; class_id++) {
    if (!AddrIsInSizeClass(addr, class_id)) continue;
    uptr size_class_first_ptr = allocated_size_classes_[class_id];
    uptr size = ClassSize(class_id);
    CHECK_LE(size_class_first_ptr, addr);
    CHECK_GT(size_class_first_ptr + ClassMmapSize(class_id), addr);
    return size_class_first_ptr + ((addr - size_class_first_ptr) / size) * size;
  }
  return 0;
}

// We may want to compute this during compilation.
ALWAYS_INLINE uptr FakeStack::ComputeSizeClass(uptr alloc_size) {
  uptr rounded_size = RoundUpToPowerOfTwo(alloc_size);
  uptr log = Log2(rounded_size);
  CHECK_LE(alloc_size, (1UL << log));
  CHECK_GT(alloc_size, (1UL << (log-1)));
  uptr res = log < kMinStackFrameSizeLog ? 0 : log - kMinStackFrameSizeLog;
  CHECK_LT(res, kNumberOfSizeClasses);
  CHECK_GE(ClassSize(res), rounded_size);
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

uptr FakeStack::ClassMmapSize(uptr class_id) {
  // Limit allocation size to ClassSize * MaxDepth when running with unlimited
  // stack.
  return RoundUpTo(Min(ClassSize(class_id) * kMaxRecursionDepth, stack_size_),
                   GetPageSizeCached());
}

void FakeStack::AllocateOneSizeClass(uptr class_id) {
  CHECK(ClassMmapSize(class_id) >= GetPageSizeCached());
  uptr new_mem = (uptr)MmapOrDie(
      ClassMmapSize(class_id), __FUNCTION__);
  if (0) {
    Printf("T%d new_mem[%zu]: %p-%p mmap %zu\n",
           GetCurrentThread()->tid(),
           class_id, new_mem, new_mem + ClassMmapSize(class_id),
           ClassMmapSize(class_id));
  }
  uptr i;
  uptr size = ClassSize(class_id);
  for (i = 0; i + size <= ClassMmapSize(class_id); i += size) {
    size_classes_[class_id].FifoPush((FakeFrame*)(new_mem + i));
  }
  CHECK_LE(i, ClassMmapSize(class_id));
  allocated_size_classes_[class_id] = new_mem;
}

ALWAYS_INLINE uptr
FakeStack::AllocateStack(uptr class_id, uptr size, uptr real_stack) {
  CHECK(size <= kMaxStackMallocSize && size > 1);
  if (!alive_) return real_stack;
  if (!allocated_size_classes_[class_id]) {
    AllocateOneSizeClass(class_id);
  }
  FakeFrame *fake_frame = size_classes_[class_id].FifoPop();
  CHECK(fake_frame);
  fake_frame->class_id = class_id;
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

ALWAYS_INLINE void FakeStack::DeallocateFrame(FakeFrame *fake_frame) {
  CHECK(alive_);
  uptr class_id = static_cast<uptr>(fake_frame->class_id);
  CHECK(allocated_size_classes_[class_id]);
  uptr ptr = (uptr)fake_frame;
  CHECK(AddrIsInSizeClass(ptr, class_id));
  size_classes_[class_id].FifoPush(fake_frame);
}

ALWAYS_INLINE void FakeStack::OnFree(uptr ptr, uptr class_id, uptr size,
                                     uptr real_stack) {
  FakeFrame *fake_frame = (FakeFrame*)ptr;
  CHECK_EQ(fake_frame->magic, kRetiredStackFrameMagic);
  CHECK_NE(fake_frame->descr, 0);
  CHECK_EQ(fake_frame->class_id, class_id);
  PoisonShadow(ptr, size, kAsanStackAfterReturnMagic);
}

ALWAYS_INLINE uptr OnMalloc(uptr class_id, uptr size, uptr real_stack) {
  if (!flags()->use_fake_stack) return real_stack;
  AsanThread *t = GetCurrentThread();
  if (!t) {
    // TSD is gone, use the real stack.
    return real_stack;
  }
  t->LazyInitFakeStack();
  uptr ptr = t->fake_stack()->AllocateStack(class_id, size, real_stack);
  // Printf("__asan_stack_malloc %p %zu %p\n", ptr, size, real_stack);
  return ptr;
}

ALWAYS_INLINE void OnFree(uptr ptr, uptr class_id, uptr size, uptr real_stack) {
  if (!flags()->use_fake_stack) return;
  if (ptr != real_stack) {
    FakeStack::OnFree(ptr, class_id, size, real_stack);
  }
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
#define DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(class_id)                       \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE uptr                                \
  __asan_stack_malloc_##class_id(uptr size, uptr real_stack) {                 \
    return __asan::OnMalloc(class_id, size, real_stack);                       \
  }                                                                            \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __asan_stack_free_##class_id(  \
      uptr ptr, uptr size, uptr real_stack) {                                  \
    __asan::OnFree(ptr, class_id, size, real_stack);                           \
  }

DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(0)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(1)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(2)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(3)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(4)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(5)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(6)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(7)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(8)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(9)
DEFINE_STACK_MALLOC_FREE_WITH_CLASS_ID(10)
