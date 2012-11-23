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

#ifdef __APPLE__

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
static malloc_zone_t *system_purgeable_zone = 0;
static malloc_zone_t asan_zone;
CFAllocatorRef cf_asan = 0;

// _CFRuntimeCreateInstance() checks whether the supplied allocator is
// kCFAllocatorSystemDefault and, if it is not, stores the allocator reference
// at the beginning of the allocated memory and returns the pointer to the
// allocated memory plus sizeof(CFAllocatorRef). See
// http://www.opensource.apple.com/source/CF/CF-635.21/CFRuntime.c
// Pointers returned by _CFRuntimeCreateInstance() can then be passed directly
// to free() or CFAllocatorDeallocate(), which leads to false invalid free
// reports.
// The corresponding rdar bug is http://openradar.appspot.com/radar?id=1796404.
void* ALWAYS_INLINE get_saved_cfallocator_ref(void *ptr) {
  if (flags()->replace_cfallocator) {
    // Make sure we're not hitting the previous page. This may be incorrect
    // if ASan's malloc returns an address ending with 0xFF8, which will be
    // then padded to a page boundary with a CFAllocatorRef.
    uptr arith_ptr = (uptr)ptr;
    if ((arith_ptr & 0xFFF) > sizeof(CFAllocatorRef)) {
      CFAllocatorRef *saved =
          (CFAllocatorRef*)(arith_ptr - sizeof(CFAllocatorRef));
      if ((*saved == cf_asan) && asan_mz_size(saved)) ptr = (void*)saved;
    }
  }
  return ptr;
}

// The free() implementation provided by OS X calls malloc_zone_from_ptr()
// to find the owner of |ptr|. If the result is 0, an invalid free() is
// reported. Our implementation falls back to asan_free() in this case
// in order to print an ASan-style report.
//
// For the objects created by _CFRuntimeCreateInstance a CFAllocatorRef is
// placed at the beginning of the allocated chunk and the pointer returned by
// our allocator is off by sizeof(CFAllocatorRef). This pointer can be then
// passed directly to free(), which will lead to errors.
// To overcome this we're checking whether |ptr-sizeof(CFAllocatorRef)|
// contains a pointer to our CFAllocator (assuming no other allocator is used).
// See http://code.google.com/p/address-sanitizer/issues/detail?id=70 for more
// info.
INTERCEPTOR(void, free, void *ptr) {
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
    if (!asan_mz_size(ptr)) ptr = get_saved_cfallocator_ref(ptr);
    GET_STACK_TRACE_HERE_FOR_FREE(ptr);
    asan_free(ptr, &stack);
  }
}

// We can't always replace the default CFAllocator with cf_asan right in
// ReplaceSystemMalloc(), because it is sometimes called before
// __CFInitialize(), when the default allocator is invalid and replacing it may
// crash the program. Instead we wait for the allocator to initialize and jump
// in just after __CFInitialize(). Nobody is going to allocate memory using
// CFAllocators before that, so we won't miss anything.
//
// See http://code.google.com/p/address-sanitizer/issues/detail?id=87
// and http://opensource.apple.com/source/CF/CF-550.43/CFRuntime.c
INTERCEPTOR(void, __CFInitialize, void) {
  // If the runtime is built as dynamic library, __CFInitialize wrapper may be
  // called before __asan_init.
#if !MAC_INTERPOSE_FUNCTIONS
  CHECK(flags()->replace_cfallocator);
  CHECK(asan_inited);
#endif
  REAL(__CFInitialize)();
  if (!cf_asan && asan_inited) MaybeReplaceCFAllocator();
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
  return asan_memalign(GetPageSizeCached(), size, &stack);
}

#define GET_ZONE_FOR_PTR(ptr) \
  malloc_zone_t *zone_ptr = malloc_zone_from_ptr(ptr); \
  const char *zone_name = (zone_ptr == 0) ? 0 : zone_ptr->zone_name

void ALWAYS_INLINE free_common(void *context, void *ptr) {
  if (!ptr) return;
  if (asan_mz_size(ptr)) {
    GET_STACK_TRACE_HERE_FOR_FREE(ptr);
    asan_free(ptr, &stack);
  } else {
    // If the pointer does not belong to any of the zones, use one of the
    // fallback methods to free memory.
    malloc_zone_t *zone_ptr = malloc_zone_from_ptr(ptr);
    if (zone_ptr == system_purgeable_zone) {
      // allocations from malloc_default_purgeable_zone() done before
      // __asan_init() may be occasionally freed via free_common().
      // see http://code.google.com/p/address-sanitizer/issues/detail?id=99.
      malloc_zone_free(zone_ptr, ptr);
    } else {
      // If the memory chunk pointer was moved to store additional
      // CFAllocatorRef, fix it back.
      ptr = get_saved_cfallocator_ref(ptr);
      GET_STACK_TRACE_HERE_FOR_FREE(ptr);
      if (!flags()->mac_ignore_invalid_free) {
        asan_free(ptr, &stack);
      } else {
        GET_ZONE_FOR_PTR(ptr);
        WarnMacFreeUnallocated((uptr)ptr, (uptr)zone_ptr, zone_name, &stack);
        return;
      }
    }
  }
}

// TODO(glider): the allocation callbacks need to be refactored.
void mz_free(malloc_zone_t *zone, void *ptr) {
  free_common(zone, ptr);
}

void cf_free(void *ptr, void *info) {
  free_common(info, ptr);
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
      GET_STACK_TRACE_HERE_FOR_FREE(ptr);
      GET_ZONE_FOR_PTR(ptr);
      ReportMacMzReallocUnknown((uptr)ptr, (uptr)zone_ptr, zone_name, &stack);
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
      GET_STACK_TRACE_HERE_FOR_FREE(ptr);
      GET_ZONE_FOR_PTR(ptr);
      ReportMacCfReallocUnknown((uptr)ptr, (uptr)zone_ptr, zone_name, &stack);
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

extern int __CFRuntimeClassTableSize;

namespace __asan {
void MaybeReplaceCFAllocator() {
  static CFAllocatorContext asan_context = {
        /*version*/ 0, /*info*/ &asan_zone,
        /*retain*/ 0, /*release*/ 0,
        /*copyDescription*/0,
        /*allocate*/ &cf_malloc,
        /*reallocate*/ &cf_realloc,
        /*deallocate*/ &cf_free,
        /*preferredSize*/ 0 };
  if (!cf_asan)
    cf_asan = CFAllocatorCreate(kCFAllocatorUseContext, &asan_context);
  if (flags()->replace_cfallocator && CFAllocatorGetDefault() != cf_asan)
    CFAllocatorSetDefault(cf_asan);
}

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

  // If __CFInitialize() hasn't been called yet, cf_asan will be created and
  // installed as the default allocator after __CFInitialize() finishes (see
  // the interceptor for __CFInitialize() above). Otherwise install cf_asan
  // right now. On both Snow Leopard and Lion __CFInitialize() calls
  // __CFAllocatorInitialize(), which initializes the _base._cfisa field of
  // the default allocators we check here.
  if (((CFRuntimeBase*)kCFAllocatorSystemDefault)->_cfisa) {
    MaybeReplaceCFAllocator();
  }
}
}  // namespace __asan

#endif  // __APPLE__
