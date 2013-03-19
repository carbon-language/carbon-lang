//===-- asan_malloc_mac.cc ------------------------------------------------===//
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
// Mac-specific malloc interception.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_MAC

#include <AvailabilityMacros.h>
#include <CoreFoundation/CFBase.h>
#include <dlfcn.h>
#include <malloc/malloc.h>

#include "asan_allocator.h"
#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_mac.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread_registry.h"

// Similar code is used in Google Perftools,
// http://code.google.com/p/google-perftools.

// ---------------------- Replacement functions ---------------- {{{1
using namespace __asan;  // NOLINT

// TODO(glider): do we need both zones?
static malloc_zone_t *system_malloc_zone = 0;
static malloc_zone_t asan_zone;

INTERCEPTOR(malloc_zone_t *, malloc_create_zone,
                             vm_size_t start_size, unsigned zone_flags) {
  if (!asan_inited) __asan_init();
  GET_STACK_TRACE_MALLOC;
  malloc_zone_t *new_zone =
      (malloc_zone_t*)asan_malloc(sizeof(asan_zone), &stack);
  internal_memcpy(new_zone, &asan_zone, sizeof(asan_zone));
  new_zone->zone_name = NULL;  // The name will be changed anyway.
  return new_zone;
}

INTERCEPTOR(malloc_zone_t *, malloc_default_zone, void) {
  if (!asan_inited) __asan_init();
  return &asan_zone;
}

INTERCEPTOR(malloc_zone_t *, malloc_default_purgeable_zone, void) {
  // FIXME: ASan should support purgeable allocations.
  // https://code.google.com/p/address-sanitizer/issues/detail?id=139
  if (!asan_inited) __asan_init();
  return &asan_zone;
}

INTERCEPTOR(void, malloc_make_purgeable, void *ptr) {
  // FIXME: ASan should support purgeable allocations. Ignoring them is fine
  // for now.
  if (!asan_inited) __asan_init();
}

INTERCEPTOR(int, malloc_make_nonpurgeable, void *ptr) {
  // FIXME: ASan should support purgeable allocations. Ignoring them is fine
  // for now.
  if (!asan_inited) __asan_init();
  // Must return 0 if the contents were not purged since the last call to
  // malloc_make_purgeable().
  return 0;
}

INTERCEPTOR(void, malloc_set_zone_name, malloc_zone_t *zone, const char *name) {
  if (!asan_inited) __asan_init();
  // Allocate |strlen("asan-") + 1 + internal_strlen(name)| bytes.
  size_t buflen = 6 + (name ? internal_strlen(name) : 0);
  InternalScopedBuffer<char> new_name(buflen);
  if (name && zone->introspect == asan_zone.introspect) {
    internal_snprintf(new_name.data(), buflen, "asan-%s", name);
    name = new_name.data();
  }

  // Call the system malloc's implementation for both external and our zones,
  // since that appropriately changes VM region protections on the zone.
  REAL(malloc_set_zone_name)(zone, name);
}

INTERCEPTOR(void *, malloc, size_t size) {
  if (!asan_inited) __asan_init();
  GET_STACK_TRACE_MALLOC;
  void *res = asan_malloc(size, &stack);
  return res;
}

INTERCEPTOR(void, free, void *ptr) {
  if (!asan_inited) __asan_init();
  if (!ptr) return;
  GET_STACK_TRACE_FREE;
  asan_free(ptr, &stack, FROM_MALLOC);
}

INTERCEPTOR(void *, realloc, void *ptr, size_t size) {
  if (!asan_inited) __asan_init();
  GET_STACK_TRACE_MALLOC;
  return asan_realloc(ptr, size, &stack);
}

INTERCEPTOR(void *, calloc, size_t nmemb, size_t size) {
  if (!asan_inited) __asan_init();
  GET_STACK_TRACE_MALLOC;
  return asan_calloc(nmemb, size, &stack);
}

INTERCEPTOR(void *, valloc, size_t size) {
  if (!asan_inited) __asan_init();
  GET_STACK_TRACE_MALLOC;
  return asan_memalign(GetPageSizeCached(), size, &stack, FROM_MALLOC);
}

INTERCEPTOR(size_t, malloc_good_size, size_t size) {
  if (!asan_inited) __asan_init();
  return asan_zone.introspect->good_size(&asan_zone, size);
}

INTERCEPTOR(int, posix_memalign, void **memptr, size_t alignment, size_t size) {
  if (!asan_inited) __asan_init();
  CHECK(memptr);
  GET_STACK_TRACE_MALLOC;
  void *result = asan_memalign(alignment, size, &stack, FROM_MALLOC);
  if (result) {
    *memptr = result;
    return 0;
  }
  return -1;
}

namespace {

// TODO(glider): the mz_* functions should be united with the Linux wrappers,
// as they are basically copied from there.
size_t mz_size(malloc_zone_t* zone, const void* ptr) {
  return asan_mz_size(ptr);
}

void *mz_malloc(malloc_zone_t *zone, size_t size) {
  if (!asan_inited) {
    CHECK(system_malloc_zone);
    return malloc_zone_malloc(system_malloc_zone, size);
  }
  GET_STACK_TRACE_MALLOC;
  return asan_malloc(size, &stack);
}

void *mz_calloc(malloc_zone_t *zone, size_t nmemb, size_t size) {
  if (!asan_inited) {
    // Hack: dlsym calls calloc before REAL(calloc) is retrieved from dlsym.
    const size_t kCallocPoolSize = 1024;
    static uptr calloc_memory_for_dlsym[kCallocPoolSize];
    static size_t allocated;
    size_t size_in_words = ((nmemb * size) + kWordSize - 1) / kWordSize;
    void *mem = (void*)&calloc_memory_for_dlsym[allocated];
    allocated += size_in_words;
    CHECK(allocated < kCallocPoolSize);
    return mem;
  }
  GET_STACK_TRACE_MALLOC;
  return asan_calloc(nmemb, size, &stack);
}

void *mz_valloc(malloc_zone_t *zone, size_t size) {
  if (!asan_inited) {
    CHECK(system_malloc_zone);
    return malloc_zone_valloc(system_malloc_zone, size);
  }
  GET_STACK_TRACE_MALLOC;
  return asan_memalign(GetPageSizeCached(), size, &stack, FROM_MALLOC);
}

#define GET_ZONE_FOR_PTR(ptr) \
  malloc_zone_t *zone_ptr = malloc_zone_from_ptr(ptr); \
  const char *zone_name = (zone_ptr == 0) ? 0 : zone_ptr->zone_name

void ALWAYS_INLINE free_common(void *context, void *ptr) {
  if (!ptr) return;
  GET_STACK_TRACE_FREE;
  // FIXME: need to retire this flag.
  if (!flags()->mac_ignore_invalid_free) {
    asan_free(ptr, &stack, FROM_MALLOC);
  } else {
    GET_ZONE_FOR_PTR(ptr);
    WarnMacFreeUnallocated((uptr)ptr, (uptr)zone_ptr, zone_name, &stack);
    return;
  }
}

// TODO(glider): the allocation callbacks need to be refactored.
void mz_free(malloc_zone_t *zone, void *ptr) {
  free_common(zone, ptr);
}

void *mz_realloc(malloc_zone_t *zone, void *ptr, size_t size) {
  if (!ptr) {
    GET_STACK_TRACE_MALLOC;
    return asan_malloc(size, &stack);
  } else {
    if (asan_mz_size(ptr)) {
      GET_STACK_TRACE_MALLOC;
      return asan_realloc(ptr, size, &stack);
    } else {
      // We can't recover from reallocating an unknown address, because
      // this would require reading at most |size| bytes from
      // potentially unaccessible memory.
      GET_STACK_TRACE_FREE;
      GET_ZONE_FOR_PTR(ptr);
      ReportMacMzReallocUnknown((uptr)ptr, (uptr)zone_ptr, zone_name, &stack);
    }
  }
}

void mz_destroy(malloc_zone_t* zone) {
  // A no-op -- we will not be destroyed!
  Report("mz_destroy() called -- ignoring\n");
}

  // from AvailabilityMacros.h
#if defined(MAC_OS_X_VERSION_10_6) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_6
void *mz_memalign(malloc_zone_t *zone, size_t align, size_t size) {
  if (!asan_inited) {
    CHECK(system_malloc_zone);
    return malloc_zone_memalign(system_malloc_zone, align, size);
  }
  GET_STACK_TRACE_MALLOC;
  return asan_memalign(align, size, &stack, FROM_MALLOC);
}

// This function is currently unused, and we build with -Werror.
#if 0
void mz_free_definite_size(malloc_zone_t* zone, void *ptr, size_t size) {
  // TODO(glider): check that |size| is valid.
  UNIMPLEMENTED();
}
#endif
#endif

kern_return_t mi_enumerator(task_t task, void *,
                            unsigned type_mask, vm_address_t zone_address,
                            memory_reader_t reader,
                            vm_range_recorder_t recorder) {
  // Should enumerate all the pointers we have.  Seems like a lot of work.
  return KERN_FAILURE;
}

size_t mi_good_size(malloc_zone_t *zone, size_t size) {
  // I think it's always safe to return size, but we maybe could do better.
  return size;
}

boolean_t mi_check(malloc_zone_t *zone) {
  UNIMPLEMENTED();
}

void mi_print(malloc_zone_t *zone, boolean_t verbose) {
  UNIMPLEMENTED();
}

void mi_log(malloc_zone_t *zone, void *address) {
  // I don't think we support anything like this
}

void mi_force_lock(malloc_zone_t *zone) {
  asan_mz_force_lock();
}

void mi_force_unlock(malloc_zone_t *zone) {
  asan_mz_force_unlock();
}

void mi_statistics(malloc_zone_t *zone, malloc_statistics_t *stats) {
  AsanMallocStats malloc_stats;
  asanThreadRegistry().FillMallocStatistics(&malloc_stats);
  CHECK(sizeof(malloc_statistics_t) == sizeof(AsanMallocStats));
  internal_memcpy(stats, &malloc_stats, sizeof(malloc_statistics_t));
}

#if defined(MAC_OS_X_VERSION_10_6) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_6
boolean_t mi_zone_locked(malloc_zone_t *zone) {
  // UNIMPLEMENTED();
  return false;
}
#endif

}  // unnamed namespace

namespace __asan {

void ReplaceSystemMalloc() {
  static malloc_introspection_t asan_introspection;
  // Ok to use internal_memset, these places are not performance-critical.
  internal_memset(&asan_introspection, 0, sizeof(asan_introspection));

  asan_introspection.enumerator = &mi_enumerator;
  asan_introspection.good_size = &mi_good_size;
  asan_introspection.check = &mi_check;
  asan_introspection.print = &mi_print;
  asan_introspection.log = &mi_log;
  asan_introspection.force_lock = &mi_force_lock;
  asan_introspection.force_unlock = &mi_force_unlock;
  asan_introspection.statistics = &mi_statistics;

  internal_memset(&asan_zone, 0, sizeof(malloc_zone_t));

  // Start with a version 4 zone which is used for OS X 10.4 and 10.5.
  asan_zone.version = 4;
  asan_zone.zone_name = "asan";
  asan_zone.size = &mz_size;
  asan_zone.malloc = &mz_malloc;
  asan_zone.calloc = &mz_calloc;
  asan_zone.valloc = &mz_valloc;
  asan_zone.free = &mz_free;
  asan_zone.realloc = &mz_realloc;
  asan_zone.destroy = &mz_destroy;
  asan_zone.batch_malloc = 0;
  asan_zone.batch_free = 0;
  asan_zone.introspect = &asan_introspection;

  // from AvailabilityMacros.h
#if defined(MAC_OS_X_VERSION_10_6) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_6
  // Switch to version 6 on OSX 10.6 to support memalign.
  asan_zone.version = 6;
  asan_zone.free_definite_size = 0;
  asan_zone.memalign = &mz_memalign;
  asan_introspection.zone_locked = &mi_zone_locked;
#endif

  // Register the ASan zone.
  malloc_zone_register(&asan_zone);
}
}  // namespace __asan

#endif  // __APPLE__
