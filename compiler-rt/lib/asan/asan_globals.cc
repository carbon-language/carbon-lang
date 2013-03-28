//===-- asan_globals.cc ---------------------------------------------------===//
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
// Handle globals.
//===----------------------------------------------------------------------===//
#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_poisoning.h"
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __asan {

typedef __asan_global Global;

struct ListOfGlobals {
  const Global *g;
  ListOfGlobals *next;
};

static BlockingMutex mu_for_globals(LINKER_INITIALIZED);
static LowLevelAllocator allocator_for_globals;
static ListOfGlobals *list_of_all_globals;

static const int kDynamicInitGlobalsInitialCapacity = 512;
typedef InternalVector<Global> VectorOfGlobals;
// Lazy-initialized and never deleted.
static VectorOfGlobals *dynamic_init_globals;

ALWAYS_INLINE void PoisonShadowForGlobal(const Global *g, u8 value) {
  FastPoisonShadow(g->beg, g->size_with_redzone, value);
}

ALWAYS_INLINE void PoisonRedZones(const Global &g) {
  uptr aligned_size = RoundUpTo(g.size, SHADOW_GRANULARITY);
  FastPoisonShadow(g.beg + aligned_size, g.size_with_redzone - aligned_size,
                   kAsanGlobalRedzoneMagic);
  if (g.size != aligned_size) {
    FastPoisonShadowPartialRightRedzone(
        g.beg + RoundDownTo(g.size, SHADOW_GRANULARITY),
        g.size % SHADOW_GRANULARITY,
        SHADOW_GRANULARITY,
        kAsanGlobalRedzoneMagic);
  }
}

static void ReportGlobal(const Global &g, const char *prefix) {
  Report("%s Global: beg=%p size=%zu/%zu name=%s module=%s dyn_init=%zu\n",
         prefix, (void*)g.beg, g.size, g.size_with_redzone, g.name,
         g.module_name, g.has_dynamic_init);
}

bool DescribeAddressIfGlobal(uptr addr, uptr size) {
  if (!flags()->report_globals) return false;
  BlockingMutexLock lock(&mu_for_globals);
  bool res = false;
  for (ListOfGlobals *l = list_of_all_globals; l; l = l->next) {
    const Global &g = *l->g;
    if (flags()->report_globals >= 2)
      ReportGlobal(g, "Search");
    res |= DescribeAddressRelativeToGlobal(addr, size, g);
  }
  return res;
}

// Register a global variable.
// This function may be called more than once for every global
// so we store the globals in a map.
static void RegisterGlobal(const Global *g) {
  CHECK(asan_inited);
  if (flags()->report_globals >= 2)
    ReportGlobal(*g, "Added");
  CHECK(flags()->report_globals);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  if (flags()->poison_heap)
    PoisonRedZones(*g);
  ListOfGlobals *l =
      (ListOfGlobals*)allocator_for_globals.Allocate(sizeof(ListOfGlobals));
  l->g = g;
  l->next = list_of_all_globals;
  list_of_all_globals = l;
  if (g->has_dynamic_init) {
    if (dynamic_init_globals == 0) {
      void *mem = allocator_for_globals.Allocate(sizeof(VectorOfGlobals));
      dynamic_init_globals = new(mem)
          VectorOfGlobals(kDynamicInitGlobalsInitialCapacity);
    }
    dynamic_init_globals->push_back(*g);
  }
}

static void UnregisterGlobal(const Global *g) {
  CHECK(asan_inited);
  CHECK(flags()->report_globals);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  if (flags()->poison_heap)
    PoisonShadowForGlobal(g, 0);
  // We unpoison the shadow memory for the global but we do not remove it from
  // the list because that would require O(n^2) time with the current list
  // implementation. It might not be worth doing anyway.
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// Register an array of globals.
void __asan_register_globals(__asan_global *globals, uptr n) {
  if (!flags()->report_globals) return;
  BlockingMutexLock lock(&mu_for_globals);
  for (uptr i = 0; i < n; i++) {
    RegisterGlobal(&globals[i]);
  }
}

// Unregister an array of globals.
// We must do this when a shared objects gets dlclosed.
void __asan_unregister_globals(__asan_global *globals, uptr n) {
  if (!flags()->report_globals) return;
  BlockingMutexLock lock(&mu_for_globals);
  for (uptr i = 0; i < n; i++) {
    UnregisterGlobal(&globals[i]);
  }
}

// This method runs immediately prior to dynamic initialization in each TU,
// when all dynamically initialized globals are unpoisoned.  This method
// poisons all global variables not defined in this TU, so that a dynamic
// initializer can only touch global variables in the same TU.
void __asan_before_dynamic_init(const char *module_name) {
  if (!flags()->check_initialization_order ||
      !flags()->poison_heap)
    return;
  CHECK(dynamic_init_globals);
  CHECK(module_name);
  CHECK(asan_inited);
  BlockingMutexLock lock(&mu_for_globals);
  if (flags()->report_globals >= 3)
    Printf("DynInitPoison module: %s\n", module_name);
  for (uptr i = 0, n = dynamic_init_globals->size(); i < n; ++i) {
    const Global *g = &(*dynamic_init_globals)[i];
    if (g->module_name != module_name)
      PoisonShadowForGlobal(g, kAsanInitializationOrderMagic);
  }
}

// This method runs immediately after dynamic initialization in each TU, when
// all dynamically initialized globals except for those defined in the current
// TU are poisoned.  It simply unpoisons all dynamically initialized globals.
void __asan_after_dynamic_init() {
  if (!flags()->check_initialization_order ||
      !flags()->poison_heap)
    return;
  CHECK(asan_inited);
  BlockingMutexLock lock(&mu_for_globals);
  // FIXME: Optionally report that we're unpoisoning globals from a module.
  for (uptr i = 0, n = dynamic_init_globals->size(); i < n; ++i) {
    const Global *g = &(*dynamic_init_globals)[i];
    // Unpoison the whole global.
    PoisonShadowForGlobal(g, 0);
    // Poison redzones back.
    PoisonRedZones(*g);
  }
}
