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

struct CovData {
  BlockingMutex mu;
  InternalMmapVector<uptr> v;
};

static uptr cov_data_placeholder[sizeof(CovData) / sizeof(uptr)];
COMPILER_CHECK(sizeof(cov_data_placeholder) >= sizeof(CovData));
static CovData *cov_data = reinterpret_cast<CovData*>(cov_data_placeholder);

namespace __sanitizer {

// Simply add the pc into the vector under lock. If the function is called more
// than once for a given PC it will be inserted multiple times, which is fine.
static void CovAdd(uptr pc) {
  BlockingMutexLock lock(&cov_data->mu);
  cov_data->v.push_back(pc);
}

static inline bool CompareLess(const uptr &a, const uptr &b) {
  return a < b;
}

// Dump the coverage on disk.
void CovDump() {
#if !SANITIZER_WINDOWS
  BlockingMutexLock lock(&cov_data->mu);
  InternalMmapVector<uptr> &v = cov_data->v;
  InternalSort(&v, v.size(), CompareLess);
  InternalMmapVector<u32> offsets(v.size());
  const uptr *vb = v.data();
  const uptr *ve = vb + v.size();
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
}  // extern "C"
