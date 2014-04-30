//===-- sanitizer_coverage.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Sanitizer Coverage.
// This file implements run-time support for a poor man's coverage tool.
//
// Compiler instrumentation:
// For every interesting basic block the compiler injects the following code:
// if (*Guard) {
//    __sanitizer_cov();
//    *Guard = 1;
// }
// It's fine to call __sanitizer_cov more than once for a given block.
//
// Run-time:
//  - __sanitizer_cov(): record that we've executed the PC (GET_CALLER_PC).
//  - __sanitizer_cov_dump: dump the coverage data to disk.
//  For every module of the current process that has coverage data
//  this will create a file module_name.PID.sancov. The file format is simple:
//  it's just a sorted sequence of 4-byte offsets in the module.
//
// Eventually, this coverage implementation should be obsoleted by a more
// powerful general purpose Clang/LLVM coverage instrumentation.
// Consider this implementation as prototype.
//
// FIXME: support (or at least test with) dlclose.
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_flags.h"

atomic_uint32_t dump_once_guard;  // Ensure that CovDump runs only once.

// pc_array is the array containing the covered PCs.
// To make the pc_array thread- and AS- safe it has to be large enough.
// 128M counters "ought to be enough for anybody" (4M on 32-bit).
// pc_array is allocated with MmapNoReserveOrDie and so it uses only as
// much RAM as it really needs.
static const uptr kPcArraySize = FIRST_32_SECOND_64(1 << 22, 1 << 27);
static uptr *pc_array;
static atomic_uintptr_t pc_array_index;

namespace __sanitizer {

// Simply add the pc into the vector under lock. If the function is called more
// than once for a given PC it will be inserted multiple times, which is fine.
static void CovAdd(uptr pc) {
  if (!pc_array) return;
  uptr idx = atomic_fetch_add(&pc_array_index, 1, memory_order_relaxed);
  CHECK_LT(idx, kPcArraySize);
  pc_array[idx] = pc;
}

void CovInit() {
  pc_array = reinterpret_cast<uptr *>(
      MmapNoReserveOrDie(sizeof(uptr) * kPcArraySize, "CovInit"));
}

static inline bool CompareLess(const uptr &a, const uptr &b) {
  return a < b;
}

// Dump the coverage on disk.
void CovDump() {
#if !SANITIZER_WINDOWS
  if (atomic_fetch_add(&dump_once_guard, 1, memory_order_relaxed))
    return;
  uptr size = atomic_load(&pc_array_index, memory_order_relaxed);
  InternalSort(&pc_array, size, CompareLess);
  InternalMmapVector<u32> offsets(size);
  const uptr *vb = pc_array;
  const uptr *ve = vb + size;
  MemoryMappingLayout proc_maps(/*cache_enabled*/false);
  uptr mb, me, off, prot;
  InternalScopedBuffer<char> module(4096);
  InternalScopedBuffer<char> path(4096 * 2);
  for (int i = 0;
       proc_maps.Next(&mb, &me, &off, module.data(), module.size(), &prot);
       i++) {
    if ((prot & MemoryMappingLayout::kProtectionExecute) == 0)
      continue;
    if (vb >= ve) break;
    if (mb <= *vb && *vb < me) {
      offsets.clear();
      const uptr *old_vb = vb;
      CHECK_LE(off, *vb);
      for (; vb < ve && *vb < me; vb++) {
        uptr diff = *vb - (i ? mb : 0) + off;
        CHECK_LE(diff, 0xffffffffU);
        offsets.push_back(static_cast<u32>(diff));
      }
      char *module_name = StripModuleName(module.data());
      internal_snprintf((char *)path.data(), path.size(), "%s.%zd.sancov",
                        module_name, internal_getpid());
      InternalFree(module_name);
      uptr fd = OpenFile(path.data(), true);
      if (internal_iserror(fd)) {
        Report(" CovDump: failed to open %s for writing\n", path.data());
      } else {
        internal_write(fd, offsets.data(), offsets.size() * sizeof(u32));
        internal_close(fd);
        VReport(1, " CovDump: %s: %zd PCs written\n", path.data(), vb - old_vb);
      }
    }
  }
#endif  // !SANITIZER_WINDOWS
}

}  // namespace __sanitizer

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov() {
  CovAdd(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_dump() { CovDump(); }
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_init() { CovInit(); }
}  // extern "C"
