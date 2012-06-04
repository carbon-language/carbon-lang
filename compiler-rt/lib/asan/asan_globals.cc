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
#include "asan_interface.h"
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_stats.h"
#include "asan_thread.h"

#include <ctype.h>

namespace __asan {

typedef __asan_global Global;

struct ListOfGlobals {
  const Global *g;
  ListOfGlobals *next;
};

static AsanLock mu_for_globals(LINKER_INITIALIZED);
static ListOfGlobals *list_of_globals;
static LowLevelAllocator allocator_for_globals(LINKER_INITIALIZED);

void PoisonRedZones(const Global &g)  {
  uptr shadow_rz_size = kGlobalAndStackRedzone >> SHADOW_SCALE;
  CHECK(shadow_rz_size == 1 || shadow_rz_size == 2 || shadow_rz_size == 4);
  // full right redzone
  uptr g_aligned_size = kGlobalAndStackRedzone *
      ((g.size + kGlobalAndStackRedzone - 1) / kGlobalAndStackRedzone);
  PoisonShadow(g.beg + g_aligned_size,
               kGlobalAndStackRedzone, kAsanGlobalRedzoneMagic);
  if ((g.size % kGlobalAndStackRedzone) != 0) {
    // partial right redzone
    u64 g_aligned_down_size = kGlobalAndStackRedzone *
        (g.size / kGlobalAndStackRedzone);
    CHECK(g_aligned_down_size == g_aligned_size - kGlobalAndStackRedzone);
    PoisonShadowPartialRightRedzone(g.beg + g_aligned_down_size,
                                    g.size % kGlobalAndStackRedzone,
                                    kGlobalAndStackRedzone,
                                    kAsanGlobalRedzoneMagic);
  }
}

static uptr GetAlignedSize(uptr size) {
  return ((size + kGlobalAndStackRedzone - 1) / kGlobalAndStackRedzone)
      * kGlobalAndStackRedzone;
}

  // Check if the global is a zero-terminated ASCII string. If so, print it.
void PrintIfASCII(const Global &g) {
  for (uptr p = g.beg; p < g.beg + g.size - 1; p++) {
    if (!isascii(*(char*)p)) return;
  }
  if (*(char*)(g.beg + g.size - 1) != 0) return;
  Printf("  '%s' is ascii string '%s'\n", g.name, g.beg);
}

bool DescribeAddrIfMyRedZone(const Global &g, uptr addr) {
  if (addr < g.beg - kGlobalAndStackRedzone) return false;
  if (addr >= g.beg + g.size_with_redzone) return false;
  Printf("%p is located ", addr);
  if (addr < g.beg) {
    Printf("%zd bytes to the left", g.beg - addr);
  } else if (addr >= g.beg + g.size) {
    Printf("%zd bytes to the right", addr - (g.beg + g.size));
  } else {
    Printf("%zd bytes inside", addr - g.beg);  // Can it happen?
  }
  Printf(" of global variable '%s' (0x%zx) of size %zu\n",
         g.name, g.beg, g.size);
  PrintIfASCII(g);
  return true;
}


bool DescribeAddrIfGlobal(uptr addr) {
  if (!FLAG_report_globals) return false;
  ScopedLock lock(&mu_for_globals);
  bool res = false;
  for (ListOfGlobals *l = list_of_globals; l; l = l->next) {
    const Global &g = *l->g;
    if (FLAG_report_globals >= 2)
      Printf("Search Global: beg=%p size=%zu name=%s\n",
             g.beg, g.size, g.name);
    res |= DescribeAddrIfMyRedZone(g, addr);
  }
  return res;
}

// Register a global variable.
// This function may be called more than once for every global
// so we store the globals in a map.
static void RegisterGlobal(const Global *g) {
  CHECK(asan_inited);
  CHECK(FLAG_report_globals);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  PoisonRedZones(*g);
  ListOfGlobals *l =
      (ListOfGlobals*)allocator_for_globals.Allocate(sizeof(ListOfGlobals));
  l->g = g;
  l->next = list_of_globals;
  list_of_globals = l;
  if (FLAG_report_globals >= 2)
    Report("Added Global: beg=%p size=%zu name=%s\n",
           g->beg, g->size, g->name);
}

static void UnregisterGlobal(const Global *g) {
  CHECK(asan_inited);
  CHECK(FLAG_report_globals);
  CHECK(AddrIsInMem(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->beg));
  CHECK(AddrIsAlignedByGranularity(g->size_with_redzone));
  PoisonShadow(g->beg, g->size_with_redzone, 0);
  // We unpoison the shadow memory for the global but we do not remove it from
  // the list because that would require O(n^2) time with the current list
  // implementation. It might not be worth doing anyway.
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// Register one global with a default redzone.
void __asan_register_global(uptr addr, uptr size,
                            const char *name) {
  if (!FLAG_report_globals) return;
  ScopedLock lock(&mu_for_globals);
  Global *g = (Global *)allocator_for_globals.Allocate(sizeof(Global));
  g->beg = addr;
  g->size = size;
  g->size_with_redzone = GetAlignedSize(size) + kGlobalAndStackRedzone;
  g->name = name;
  RegisterGlobal(g);
}

// Register an array of globals.
void __asan_register_globals(__asan_global *globals, uptr n) {
  if (!FLAG_report_globals) return;
  ScopedLock lock(&mu_for_globals);
  for (uptr i = 0; i < n; i++) {
    RegisterGlobal(&globals[i]);
  }
}

// Unregister an array of globals.
// We must do it when a shared objects gets dlclosed.
void __asan_unregister_globals(__asan_global *globals, uptr n) {
  if (!FLAG_report_globals) return;
  ScopedLock lock(&mu_for_globals);
  for (uptr i = 0; i < n; i++) {
    UnregisterGlobal(&globals[i]);
  }
}
