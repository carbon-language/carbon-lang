//===-- tsan_platform_posix.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// POSIX-specific code.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"

namespace __tsan {

#ifndef SANITIZER_GO
void InitializeShadowMemory() {
  // Map memory shadow.
  uptr shadow =
      (uptr)MmapFixedNoReserve(kShadowBeg, kShadowEnd - kShadowBeg, "shadow");
  if (shadow != kShadowBeg) {
    Printf("FATAL: ThreadSanitizer can not mmap the shadow memory\n");
    Printf("FATAL: Make sure to compile with -fPIE and "
               "to link with -pie (%p, %p).\n", shadow, kShadowBeg);
    Die();
  }
  // This memory range is used for thread stacks and large user mmaps.
  // Frequently a thread uses only a small part of stack and similarly
  // a program uses a small part of large mmap. On some programs
  // we see 20% memory usage reduction without huge pages for this range.
  // FIXME: don't use constants here.
#if defined(__x86_64__)
  const uptr kMadviseRangeBeg  = 0x7f0000000000ull;
  const uptr kMadviseRangeSize = 0x010000000000ull;
#elif defined(__mips64)
  const uptr kMadviseRangeBeg  = 0xff00000000ull;
  const uptr kMadviseRangeSize = 0x0100000000ull;
#elif defined(__aarch64__)
  const uptr kMadviseRangeBeg  = 0x7e00000000ull;
  const uptr kMadviseRangeSize = 0x0100000000ull;
#endif
  NoHugePagesInRegion(MemToShadow(kMadviseRangeBeg),
                      kMadviseRangeSize * kShadowMultiplier);
  // Meta shadow is compressing and we don't flush it,
  // so it makes sense to mark it as NOHUGEPAGE to not over-allocate memory.
  // On one program it reduces memory consumption from 5GB to 2.5GB.
  NoHugePagesInRegion(kMetaShadowBeg, kMetaShadowEnd - kMetaShadowBeg);
  if (common_flags()->use_madv_dontdump)
    DontDumpShadowMemory(kShadowBeg, kShadowEnd - kShadowBeg);
  DPrintf("memory shadow: %zx-%zx (%zuGB)\n",
      kShadowBeg, kShadowEnd,
      (kShadowEnd - kShadowBeg) >> 30);

  // Map meta shadow.
  uptr meta_size = kMetaShadowEnd - kMetaShadowBeg;
  uptr meta =
      (uptr)MmapFixedNoReserve(kMetaShadowBeg, meta_size, "meta shadow");
  if (meta != kMetaShadowBeg) {
    Printf("FATAL: ThreadSanitizer can not mmap the shadow memory\n");
    Printf("FATAL: Make sure to compile with -fPIE and "
               "to link with -pie (%p, %p).\n", meta, kMetaShadowBeg);
    Die();
  }
  if (common_flags()->use_madv_dontdump)
    DontDumpShadowMemory(meta, meta_size);
  DPrintf("meta shadow: %zx-%zx (%zuGB)\n",
      meta, meta + meta_size, meta_size >> 30);

  InitializeShadowMemoryPlatform();
}

static void ProtectRange(uptr beg, uptr end) {
  CHECK_LE(beg, end);
  if (beg == end)
    return;
  if (beg != (uptr)MmapNoAccess(beg, end - beg)) {
    Printf("FATAL: ThreadSanitizer can not protect [%zx,%zx]\n", beg, end);
    Printf("FATAL: Make sure you are not using unlimited stack\n");
    Die();
  }
}

void CheckAndProtect() {
  // Ensure that the binary is indeed compiled with -pie.
  MemoryMappingLayout proc_maps(true);
  uptr p, end, prot;
  while (proc_maps.Next(&p, &end, 0, 0, 0, &prot)) {
    if (IsAppMem(p))
      continue;
    if (p >= kHeapMemEnd &&
        p < HeapEnd())
      continue;
    if (prot == 0)  // Zero page or mprotected.
      continue;
    if (p >= kVdsoBeg)  // vdso
      break;
    Printf("FATAL: ThreadSanitizer: unexpected memory mapping %p-%p\n", p, end);
    Die();
  }

  ProtectRange(kLoAppMemEnd, kShadowBeg);
  ProtectRange(kShadowEnd, kMetaShadowBeg);
  ProtectRange(kMetaShadowEnd, kTraceMemBeg);
  // Memory for traces is mapped lazily in MapThreadTrace.
  // Protect the whole range for now, so that user does not map something here.
  ProtectRange(kTraceMemBeg, kTraceMemEnd);
  ProtectRange(kTraceMemEnd, kHeapMemBeg);
  ProtectRange(HeapEnd(), kHiAppMemBeg);
}
#endif

}  // namespace __tsan

#endif  // SANITIZER_POSIX
