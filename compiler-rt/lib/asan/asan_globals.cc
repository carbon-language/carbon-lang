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
#include "asan_report.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_mutex.h"

namespace __asan {

typedef __asan_global Global;

struct ListOfGlobals {
  const Global *g;
  ListOfGlobals *next;
};

static BlockingMutex mu_for_globals(LINKER_INITIALIZED);
static LowLevelAllocator allocator_for_globals;
static ListOfGlobals *list_of_all_globals;
static ListOfGlobals *list_of_dynamic_init_globals;

static void PoisonRedZones(const Global &g) {
  uptr aligned_size = RoundUpTo(g.size, SHADOW_GRANULARITY);
  PoisonShadow(g.beg + aligned_size, g.size_with_redzone - aligned_size,
               kAsanGlobalRedzoneMagic);
  if (g.size != aligned_size) {
    // partial right redzone
    PoisonShadowPartialRightRedzone(
        g.beg + RoundDownTo(g.size, SHADOW_GRANULARITY),
        g.size % SHADOW_GRANULARITY,
        SHADOW_GRANULARITY,
        kAsanGlobalRedzoneMagic);
  }
}

bool DescribeAddressIfGlobal(uptr addr, uptr size) {
  if (!flags()->report_globals) return false;
  BlockingMutexLock lock(&mu_for_globals);
  bool res = false;
  for (ListOfGlobals *l = list_of_all_globals; l; l = l->next) {
    const Global &g = *l->g;
    if (flags()->report_globals >= 2)
      Report("Search Global: beg=%p size=%zu name=%s\n",
             (void*)g.beg, g.size, (char*)g.name);
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
    Report("Added Global: beg=%p size=%zu/%zu name=%s dyn.init=%zu\n",
           (void*)g->beg, g->size, g->size_with_redzone, g->name,
           g->has_dynamic_init);
  CHECK(flags()->report_globals);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  PoisonRedZones(*g);
  ListOfGlobals *l =
      (ListOfGlobals*)allocator_for_globals.Allocate(sizeof(ListOfGlobals));
  l->g = g;
  l->next = list_of_all_globals;
  list_of_all_globals = l;
  if (g->has_dynamic_init) {
    l = (ListOfGlobals*)allocator_for_globals.Allocate(sizeof(ListOfGlobals));
    l->g = g;
    l->next = list_of_dynamic_init_globals;
    list_of_dynamic_init_globals = l;
  }
}

static void UnregisterGlobal(const Global *g) {
  CHECK(asan_inited);
  CHECK(flags()->report_globals);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  PoisonShadow(g->beg, g->size_with_redzone, 0);
  // We unpoison the shadow memory for the global but we do not remove it from
  // the list because that would require O(n^2) time with the current list
  // implementation. It might not be worth doing anyway.
}

// Poison all shadow memory for a single global.
static void PoisonGlobalAndRedzones(const Global *g) {
  CHECK(asan_inited);
  CHECK(flags()->check_initialization_order);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  if (flags()->report_globals >= 3)
    Printf("DynInitPoison  : %s\n", g->name);
  PoisonShadow(g->beg, g->size_with_redzone, kAsanInitializationOrderMagic);
}

static void UnpoisonGlobal(const Global *g) {
  CHECK(asan_inited);
  CHECK(flags()->check_initialization_order);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  if (flags()->report_globals >= 3)
    Printf("DynInitUnpoison: %s\n", g->name);
  PoisonShadow(g->beg, g->size_with_redzone, 0);
  PoisonRedZones(*g);
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
void __asan_before_dynamic_init(uptr first_addr, uptr last_addr) {
  if (!flags()->check_initialization_order) return;
  CHECK(list_of_dynamic_init_globals);
  BlockingMutexLock lock(&mu_for_globals);
  bool from_current_tu = false;
  // The list looks like:
  // a => ... => b => last_addr => ... => first_addr => c => ...
  // The globals of the current TU reside between last_addr and first_addr.
  for (ListOfGlobals *l = list_of_dynamic_init_globals; l; l = l->next) {
    if (l->g->beg == last_addr)
      from_current_tu = true;
    if (!from_current_tu)
      PoisonGlobalAndRedzones(l->g);
    if (l->g->beg == first_addr)
      from_current_tu = false;
  }
  CHECK(!from_current_tu);
}

// This method runs immediately after dynamic initialization in each TU, when
// all dynamically initialized globals except for those defined in the current
// TU are poisoned.  It simply unpoisons all dynamically initialized globals.
void __asan_after_dynamic_init() {
  if (!flags()->check_initialization_order) return;
  BlockingMutexLock lock(&mu_for_globals);
  for (ListOfGlobals *l = list_of_dynamic_init_globals; l; l = l->next)
    UnpoisonGlobal(l->g);
}
