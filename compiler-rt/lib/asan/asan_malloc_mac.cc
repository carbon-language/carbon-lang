//===-- asan_rtl.cc -------------------------------------------------------===//
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

#ifdef __APPLE__

#include <AvailabilityMacros.h>
#include <CoreFoundation/CFBase.h>
#include <malloc/malloc.h>
#include <setjmp.h>

#include "asan_allocator.h"
#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_stack.h"

// Similar code is used in Google Perftools,
// http://code.google.com/p/google-perftools.

// ---------------------- Replacement functions ---------------- {{{1
using namespace __asan;  // NOLINT

// The free() implementation provided by OS X calls malloc_zone_from_ptr()
// to find the owner of |ptr|. If the result is 0, an invalid free() is
// reported. Our implementation falls back to asan_free() in this case
// in order to print an ASan-style report.
extern "C"
void free(void *ptr) {
  malloc_zone_t *zone = malloc_zone_from_ptr(ptr);
  if (zone) {
#if defined(MAC_OS_X_VERSION_10_6) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_6
    if ((zone->version >= 6) && (zone->free_definite_size)) {
      zone->free_definite_size(zone, ptr, malloc_size(ptr));
    } else {
      malloc_zone_free(zone, ptr);
    }
#else
    malloc_zone_free(zone, ptr);
#endif
  } else {
    GET_STACK_TRACE_HERE_FOR_FREE(ptr);
    asan_free(ptr, &stack);
  }
}

// TODO(glider): do we need both zones?
static malloc_zone_t *system_malloc_zone = 0;
static malloc_zone_t *system_purgeable_zone = 0;

// We need to provide wrappers around all the libc functions.
namespace {
// TODO(glider): the mz_* functions should be united with the Linux wrappers,
// as they are basically copied from there.
size_t mz_size(malloc_zone_t* zone, const void* ptr) {
  // Fast path: check whether this pointer belongs to the original malloc zone.
  // We cannot just call malloc_zone_from_ptr(), because it in turn
  // calls our mz_size().
  if (system_malloc_zone) {
    if ((system_malloc_zone->size)(system_malloc_zone, ptr)) return 0;
  }
  return asan_mz_size(ptr);
}

void *mz_malloc(malloc_zone_t *zone, size_t size) {
  if (!asan_inited) {
    CHECK(system_malloc_zone);
    return malloc_zone_malloc(system_malloc_zone, size);
  }
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_malloc(size, &stack);
}

void *cf_malloc(CFIndex size, CFOptionFlags hint, void *info) {
  if (!asan_inited) {
    CHECK(system_malloc_zone);
    return malloc_zone_malloc(system_malloc_zone, size);
  }
  GET_STACK_TRACE_HERE_FOR_MALLOC;
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
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_calloc(nmemb, size, &stack);
}

void *mz_valloc(malloc_zone_t *zone, size_t size) {
  if (!asan_inited) {
    CHECK(system_malloc_zone);
    return malloc_zone_valloc(system_malloc_zone, size);
  }
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_memalign(kPageSize, size, &stack);
}

void print_zone_for_ptr(void *ptr) {
  malloc_zone_t *orig_zone = malloc_zone_from_ptr(ptr);
  if (orig_zone) {
    if (orig_zone->zone_name) {
      Printf("malloc_zone_from_ptr(%p) = %p, which is %s\n",
             ptr, orig_zone, orig_zone->zone_name);
    } else {
      Printf("malloc_zone_from_ptr(%p) = %p, which doesn't have a name\n",
             ptr, orig_zone);
    }
  } else {
    Printf("malloc_zone_from_ptr(%p) = 0\n", ptr);
  }
}

// TODO(glider): the allocation callbacks need to be refactored.
void mz_free(malloc_zone_t *zone, void *ptr) {
  if (!ptr) return;
  malloc_zone_t *orig_zone = malloc_zone_from_ptr(ptr);
  // For some reason Chromium calls mz_free() for pointers that belong to
  // DefaultPurgeableMallocZone instead of asan_zone. We might want to
  // fix this someday.
  if (orig_zone == system_purgeable_zone) {
    system_purgeable_zone->free(system_purgeable_zone, ptr);
    return;
  }
  if (asan_mz_size(ptr)) {
    GET_STACK_TRACE_HERE_FOR_FREE(ptr);
    asan_free(ptr, &stack);
  } else {
    // Let us just leak this memory for now.
    Printf("mz_free(%p) -- attempting to free unallocated memory.\n"
           "AddressSanitizer is ignoring this error on Mac OS now.\n", ptr);
    print_zone_for_ptr(ptr);
    GET_STACK_TRACE_HERE_FOR_FREE(ptr);
    stack.PrintStack();
    return;
  }
}

void cf_free(void *ptr, void *info) {
  if (!ptr) return;
  malloc_zone_t *orig_zone = malloc_zone_from_ptr(ptr);
  // For some reason Chromium calls mz_free() for pointers that belong to
  // DefaultPurgeableMallocZone instead of asan_zone. We might want to
  // fix this someday.
  if (orig_zone == system_purgeable_zone) {
    system_purgeable_zone->free(system_purgeable_zone, ptr);
    return;
  }
  if (asan_mz_size(ptr)) {
    GET_STACK_TRACE_HERE_FOR_FREE(ptr);
    asan_free(ptr, &stack);
  } else {
    // Let us just leak this memory for now.
    Printf("cf_free(%p) -- attempting to free unallocated memory.\n"
           "AddressSanitizer is ignoring this error on Mac OS now.\n", ptr);
    print_zone_for_ptr(ptr);
    GET_STACK_TRACE_HERE_FOR_FREE(ptr);
    stack.PrintStack();
    return;
  }
}

void *mz_realloc(malloc_zone_t *zone, void *ptr, size_t size) {
  if (!ptr) {
    GET_STACK_TRACE_HERE_FOR_MALLOC;
    return asan_malloc(size, &stack);
  } else {
    if (asan_mz_size(ptr)) {
      GET_STACK_TRACE_HERE_FOR_MALLOC;
      return asan_realloc(ptr, size, &stack);
    } else {
      // We can't recover from reallocating an unknown address, because
      // this would require reading at most |size| bytes from
      // potentially unaccessible memory.
      Printf("mz_realloc(%p) -- attempting to realloc unallocated memory.\n"
             "This is an unrecoverable problem, exiting now.\n", ptr);
      print_zone_for_ptr(ptr);
      GET_STACK_TRACE_HERE_FOR_FREE(ptr);
      stack.PrintStack();
      ShowStatsAndAbort();
      return 0;  // unreachable
    }
  }
}

void *cf_realloc(void *ptr, CFIndex size, CFOptionFlags hint, void *info) {
  if (!ptr) {
    GET_STACK_TRACE_HERE_FOR_MALLOC;
    return asan_malloc(size, &stack);
  } else {
    if (asan_mz_size(ptr)) {
      GET_STACK_TRACE_HERE_FOR_MALLOC;
      return asan_realloc(ptr, size, &stack);
    } else {
      // We can't recover from reallocating an unknown address, because
      // this would require reading at most |size| bytes from
      // potentially unaccessible memory.
      Printf("cf_realloc(%p) -- attempting to realloc unallocated memory.\n"
             "This is an unrecoverable problem, exiting now.\n", ptr);
      print_zone_for_ptr(ptr);
      GET_STACK_TRACE_HERE_FOR_FREE(ptr);
      stack.PrintStack();
      ShowStatsAndAbort();
      return 0;  // unreachable
    }
  }
}

void mz_destroy(malloc_zone_t* zone) {
  // A no-op -- we will not be destroyed!
  Printf("mz_destroy() called -- ignoring\n");
}
  // from AvailabilityMacros.h
#if defined(MAC_OS_X_VERSION_10_6) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_6
void *mz_memalign(malloc_zone_t *zone, size_t align, size_t size) {
  if (!asan_inited) {
    CHECK(system_malloc_zone);
    return malloc_zone_memalign(system_malloc_zone, align, size);
  }
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_memalign(align, size, &stack);
}

// This function is currently unused, and we build with -Werror.
#if 0
void mz_free_definite_size(malloc_zone_t* zone, void *ptr, size_t size) {
  // TODO(glider): check that |size| is valid.
  UNIMPLEMENTED();
}
#endif
#endif

// malloc_introspection callbacks.  I'm not clear on what all of these do.
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
  return true;
}

void mi_print(malloc_zone_t *zone, boolean_t verbose) {
  UNIMPLEMENTED();
  return;
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

// This function is currently unused, and we build with -Werror.
#if 0
void mi_statistics(malloc_zone_t *zone, malloc_statistics_t *stats) {
  // TODO(csilvers): figure out how to fill these out
  // TODO(glider): port this from tcmalloc when ready.
  stats->blocks_in_use = 0;
  stats->size_in_use = 0;
  stats->max_size_in_use = 0;
  stats->size_allocated = 0;
}
#endif

#if defined(MAC_OS_X_VERSION_10_6) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_6
boolean_t mi_zone_locked(malloc_zone_t *zone) {
  // UNIMPLEMENTED();
  return false;
}
#endif

}  // unnamed namespace

extern bool kCFUseCollectableAllocator;  // is GC on?

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

  static malloc_zone_t asan_zone;
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

  // Request the default purgable zone to force its creation. The
  // current default zone is registered with the purgable zone for
  // doing tiny and small allocs.  Sadly, it assumes that the default
  // zone is the szone implementation from OS X and will crash if it
  // isn't.  By creating the zone now, this will be true and changing
  // the default zone won't cause a problem.  (OS X 10.6 and higher.)
  system_purgeable_zone = malloc_default_purgeable_zone();
#endif

  // Register the ASan zone. At this point, it will not be the
  // default zone.
  malloc_zone_register(&asan_zone);

  // Unregister and reregister the default zone.  Unregistering swaps
  // the specified zone with the last one registered which for the
  // default zone makes the more recently registered zone the default
  // zone.  The default zone is then re-registered to ensure that
  // allocations made from it earlier will be handled correctly.
  // Things are not guaranteed to work that way, but it's how they work now.
  system_malloc_zone = malloc_default_zone();
  malloc_zone_unregister(system_malloc_zone);
  malloc_zone_register(system_malloc_zone);
  // Make sure the default allocator was replaced.
  CHECK(malloc_default_zone() == &asan_zone);

  if (FLAG_replace_cfallocator) {
    static CFAllocatorContext asan_context =
        { /*version*/ 0, /*info*/ &asan_zone,
          /*retain*/ 0, /*release*/ 0,
          /*copyDescription*/0,
          /*allocate*/ &cf_malloc,
          /*reallocate*/ &cf_realloc,
          /*deallocate*/ &cf_free,
          /*preferredSize*/ 0 };
    CFAllocatorRef cf_asan =
        CFAllocatorCreate(kCFAllocatorUseContext, &asan_context);
    CFAllocatorSetDefault(cf_asan);
  }
}
}  // namespace __asan

#endif  // __APPLE__
