//=-- lsan_common_linux.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Implementation of common leak checking functionality. Linux-specific code.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#include "lsan_common.h"

#if CAN_SANITIZE_LEAKS && SANITIZER_LINUX
#include <link.h>

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __lsan {

static const char kLinkerName[] = "ld";
// We request 2 modules matching "ld", so we can print a warning if there's more
// than one match. But only the first one is actually used.
static char linker_placeholder[2 * sizeof(LoadedModule)] ALIGNED(64);
static LoadedModule *linker = 0;

static bool IsLinker(const char* full_name) {
  return LibraryNameIs(full_name, kLinkerName);
}

void InitializePlatformSpecificModules() {
  internal_memset(linker_placeholder, 0, sizeof(linker_placeholder));
  uptr num_matches = GetListOfModules(
      reinterpret_cast<LoadedModule *>(linker_placeholder), 2, IsLinker);
  if (num_matches == 1) {
    linker = reinterpret_cast<LoadedModule *>(linker_placeholder);
    return;
  }
  if (num_matches == 0)
    Report("LeakSanitizer: Dynamic linker not found. "
           "TLS will not be handled correctly.\n");
  else if (num_matches > 1)
    Report("LeakSanitizer: Multiple modules match \"%s\". "
           "TLS will not be handled correctly.\n", kLinkerName);
  linker = 0;
}

static int ProcessGlobalRegionsCallback(struct dl_phdr_info *info, size_t size,
                                        void *data) {
  Frontier *frontier = reinterpret_cast<Frontier *>(data);
  for (uptr j = 0; j < info->dlpi_phnum; j++) {
    const ElfW(Phdr) *phdr = &(info->dlpi_phdr[j]);
    // We're looking for .data and .bss sections, which reside in writeable,
    // loadable segments.
    if (!(phdr->p_flags & PF_W) || (phdr->p_type != PT_LOAD) ||
        (phdr->p_memsz == 0))
      continue;
    uptr begin = info->dlpi_addr + phdr->p_vaddr;
    uptr end = begin + phdr->p_memsz;
    uptr allocator_begin = 0, allocator_end = 0;
    GetAllocatorGlobalRange(&allocator_begin, &allocator_end);
    if (begin <= allocator_begin && allocator_begin < end) {
      CHECK_LE(allocator_begin, allocator_end);
      CHECK_LT(allocator_end, end);
      if (begin < allocator_begin)
        ScanRangeForPointers(begin, allocator_begin, frontier, "GLOBAL",
                             kReachable);
      if (allocator_end < end)
        ScanRangeForPointers(allocator_end, end, frontier, "GLOBAL",
                             kReachable);
    } else {
      ScanRangeForPointers(begin, end, frontier, "GLOBAL", kReachable);
    }
  }
  return 0;
}

// Scans global variables for heap pointers.
void ProcessGlobalRegions(Frontier *frontier) {
  // FIXME: dl_iterate_phdr acquires a linker lock, so we run a risk of
  // deadlocking by running this under StopTheWorld. However, the lock is
  // reentrant, so we should be able to fix this by acquiring the lock before
  // suspending threads.
  dl_iterate_phdr(ProcessGlobalRegionsCallback, frontier);
}

static uptr GetCallerPC(u32 stack_id) {
  CHECK(stack_id);
  uptr size = 0;
  const uptr *trace = StackDepotGet(stack_id, &size);
  // The top frame is our malloc/calloc/etc. The next frame is the caller.
  if (size >= 2)
    return trace[1];
  return 0;
}

// ForEachChunk callback. Identifies unreachable chunks which must be treated as
// reachable. Marks them as reachable and adds them to the frontier.
static void ProcessPlatformSpecificAllocationsCb(uptr chunk, void *arg) {
  CHECK(arg);
  chunk = GetUserBegin(chunk);
  LsanMetadata m(chunk);
  if (m.allocated() && m.tag() != kReachable) {
    if (linker->containsAddress(GetCallerPC(m.stack_trace_id()))) {
      m.set_tag(kReachable);
      reinterpret_cast<Frontier *>(arg)->push_back(chunk);
    }
  }
}

// Handles dynamically allocated TLS blocks by treating all chunks allocated
// from ld-linux.so as reachable.
void ProcessPlatformSpecificAllocations(Frontier *frontier) {
  if (!flags()->use_tls) return;
  if (!linker) return;
  ForEachChunk(ProcessPlatformSpecificAllocationsCb, frontier);
}

}  // namespace __lsan
#endif  // CAN_SANITIZE_LEAKS && SANITIZER_LINUX
