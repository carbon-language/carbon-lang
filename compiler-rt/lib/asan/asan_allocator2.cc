//===-- asan_allocator2.cc ------------------------------------------------===//
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
// Implementation of ASan's memory allocator, 2-nd version.
// This variant uses the allocator from sanitizer_common, i.e. the one shared
// with ThreadSanitizer and MemorySanitizer.
//
// Status: under development, not enabled by default yet.
//===----------------------------------------------------------------------===//
#include "asan_allocator.h"
#if ASAN_ALLOCATOR_VERSION == 2

#include "sanitizer/asan_interface.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __asan {

#if SANITIZER_WORDSIZE == 64
const uptr kAllocatorSpace = 0x600000000000ULL;
const uptr kAllocatorSize  =  0x10000000000ULL;  // 1T.
typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize, 0 /*metadata*/,
    DefaultSizeClassMap> PrimaryAllocator;
#elif SANITIZER_WORDSIZE == 32
static const u64 kAddressSpaceSize = 1ULL << 32;
typedef SizeClassAllocator32<
  0, kAddressSpaceSize, 16, CompactSizeClassMap> PrimaryAllocator;
#endif

typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
    SecondaryAllocator> Allocator;


uptr AsanChunkView::Beg() { return 0; }
uptr AsanChunkView::End() { return Beg() + UsedSize(); }
uptr AsanChunkView::UsedSize() { return 0; }
uptr AsanChunkView::AllocTid() { return 0; }
uptr AsanChunkView::FreeTid() { return 0; }

void AsanChunkView::GetAllocStack(StackTrace *stack) { }
void AsanChunkView::GetFreeStack(StackTrace *stack) { }
AsanChunkView FindHeapChunkByAddress(uptr address) {
  UNIMPLEMENTED();
  return AsanChunkView(0);
}

void AsanThreadLocalMallocStorage::CommitBack() {
  UNIMPLEMENTED();
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_memalign(uptr alignment, uptr size, StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
}

SANITIZER_INTERFACE_ATTRIBUTE
void asan_free(void *ptr, StackTrace *stack) {
  UNIMPLEMENTED();
  return;
}

SANITIZER_INTERFACE_ATTRIBUTE
void *asan_malloc(uptr size, StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
}

void *asan_calloc(uptr nmemb, uptr size, StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
}

void *asan_realloc(void *p, uptr size, StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
}

void *asan_valloc(uptr size, StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
}

void *asan_pvalloc(uptr size, StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
}

int asan_posix_memalign(void **memptr, uptr alignment, uptr size,
                          StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
}

uptr asan_malloc_usable_size(void *ptr, StackTrace *stack) {
  UNIMPLEMENTED();
  return 0;
}

uptr asan_mz_size(const void *ptr) {
  UNIMPLEMENTED();
  return 0;
}

void asan_mz_force_lock() {
  UNIMPLEMENTED();
}

void asan_mz_force_unlock() {
  UNIMPLEMENTED();
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// ASan allocator doesn't reserve extra bytes, so normally we would
// just return "size".
uptr __asan_get_estimated_allocated_size(uptr size) {
  UNIMPLEMENTED();
  return 0;
}

bool __asan_get_ownership(const void *p) {
  UNIMPLEMENTED();
  return false;
}

uptr __asan_get_allocated_size(const void *p) {
  UNIMPLEMENTED();
  return 0;
}

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
// Provide default (no-op) implementation of malloc hooks.
extern "C" {
SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE
void __asan_malloc_hook(void *ptr, uptr size) {
  (void)ptr;
  (void)size;
}
SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE
void __asan_free_hook(void *ptr) {
  (void)ptr;
}
}  // extern "C"
#endif


#endif  // ASAN_ALLOCATOR_VERSION
