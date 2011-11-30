//===-- asan_globals.cc -----------------------------------------*- C++ -*-===//
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
#include <map>

namespace __asan {

typedef __asan_global Global;

static AsanLock mu_for_globals(LINKER_INITIALIZED);
typedef std::map<uintptr_t, Global> MapOfGlobals;
static MapOfGlobals *g_all_globals = NULL;

void PoisonRedZones(const Global &g)  {
  size_t shadow_rz_size = kGlobalAndStackRedzone >> SHADOW_SCALE;
  CHECK(shadow_rz_size == 1 || shadow_rz_size == 2 || shadow_rz_size == 4);
  // full right redzone
  size_t g_aligned_size = kGlobalAndStackRedzone *
      ((g.size + kGlobalAndStackRedzone - 1) / kGlobalAndStackRedzone);
  PoisonShadow(g.beg + g_aligned_size,
               kGlobalAndStackRedzone, kAsanGlobalRedzoneMagic);
  if ((g.size % kGlobalAndStackRedzone) != 0) {
    // partial right redzone
    uint64_t g_aligned_down_size = kGlobalAndStackRedzone *
        (g.size / kGlobalAndStackRedzone);
    CHECK(g_aligned_down_size == g_aligned_size - kGlobalAndStackRedzone);
    PoisonShadowPartialRightRedzone(g.beg + g_aligned_down_size,
                                    g.size % kGlobalAndStackRedzone,
                                    kGlobalAndStackRedzone,
                                    kAsanGlobalRedzoneMagic);
  }
}

static size_t GetAlignedSize(size_t size) {
  return ((size + kGlobalAndStackRedzone - 1) / kGlobalAndStackRedzone)
      * kGlobalAndStackRedzone;
}

  // Check if the global is a zero-terminated ASCII string. If so, print it.
void PrintIfASCII(const Global &g) {
  for (size_t p = g.beg; p < g.beg + g.size - 1; p++) {
    if (!isascii(*(char*)p)) return;
  }
  if (*(char*)(g.beg + g.size - 1) != 0) return;
  Printf("  '%s' is ascii string '%s'\n", g.name, g.beg);
}

bool DescribeAddrIfMyRedZone(const Global &g, uintptr_t addr) {
  if (addr < g.beg - kGlobalAndStackRedzone) return false;
  if (addr >= g.beg + g.size_with_redzone) return false;
  Printf("%p is located ", addr);
  if (addr < g.beg) {
    Printf("%d bytes to the left", g.beg - addr);
  } else if (addr >= g.beg + g.size) {
    Printf("%d bytes to the right", addr - (g.beg + g.size));
  } else {
    Printf("%d bytes inside", addr - g.beg);  // Can it happen?
  }
  Printf(" of global variable '%s' (0x%lx) of size %ld\n",
         g.name, g.beg, g.size);
  PrintIfASCII(g);
  return true;
}


bool DescribeAddrIfGlobal(uintptr_t addr) {
  if (!FLAG_report_globals) return false;
  ScopedLock lock(&mu_for_globals);
  if (!g_all_globals) return false;
  bool res = false;
  // Just iterate. May want to use binary search instead.
  for (MapOfGlobals::iterator i = g_all_globals->begin(),
       end = g_all_globals->end(); i != end; ++i) {
    Global &g = i->second;
    CHECK(i->first == g.beg);
    if (FLAG_report_globals >= 2)
      Printf("Search Global: beg=%p size=%ld name=%s\n",
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
  if (!FLAG_report_globals) return;
  ScopedLock lock(&mu_for_globals);
  if (!g_all_globals)
    g_all_globals = new MapOfGlobals;
  CHECK(AddrIsInMem(g->beg));
  if (FLAG_report_globals >= 2)
    Printf("Added Global: beg=%p size=%ld name=%s\n",
           g->beg, g->size, g->name);
  CHECK(AddrIsAlignedByGranularity(g->beg));
  PoisonRedZones(*g);
  (*g_all_globals)[g->beg] = *g;
}

}  // namespace __asan

// ---------------------- Interface ---------------- {{{1
using namespace __asan;  // NOLINT

// Register one global with a default redzone.
void __asan_register_global(uintptr_t addr, size_t size,
                            const char *name) {
  Global g;
  g.beg = addr;
  g.size = size;
  g.size_with_redzone = GetAlignedSize(size) + kGlobalAndStackRedzone;
  g.name = name;
  RegisterGlobal(&g);
}

// Register an array of globals.
void __asan_register_globals(__asan_global *globals, size_t n) {
  for (size_t i = 0; i < n; i++) {
    RegisterGlobal(&globals[i]);
  }
}
