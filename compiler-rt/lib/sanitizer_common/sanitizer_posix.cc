//===-- sanitizer_posix.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements POSIX-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//
#if defined(__linux__) || defined(__APPLE__)

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_procmaps.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef ANDROID
#include <sys/atomics.h>
#endif

namespace __sanitizer {

// ------------- sanitizer_common.h

int GetPid() {
  return getpid();
}

uptr GetThreadSelf() {
  return (uptr)pthread_self();
}

void *MmapOrDie(uptr size, const char *mem_type) {
  size = RoundUpTo(size, kPageSize);
  void *res = internal_mmap(0, size,
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANON, -1, 0);
  if (res == (void*)-1) {
    Report("ERROR: Failed to allocate 0x%zx (%zd) bytes of %s\n",
           size, size, mem_type);
    CHECK("unable to mmap" && 0);
  }
  return res;
}

void UnmapOrDie(void *addr, uptr size) {
  if (!addr || !size) return;
  int res = internal_munmap(addr, size);
  if (res != 0) {
    Report("ERROR: Failed to deallocate 0x%zx (%zd) bytes at address %p\n",
           size, size, addr);
    CHECK("unable to unmap" && 0);
  }
}

void *MmapFixedNoReserve(uptr fixed_addr, uptr size) {
  return internal_mmap((void*)fixed_addr, size,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
                      -1, 0);
}

void *Mprotect(uptr fixed_addr, uptr size) {
  return internal_mmap((void*)fixed_addr, size,
                       PROT_NONE,
                       MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
                       -1, 0);
}

static inline bool IntervalsAreSeparate(uptr start1, uptr end1,
                                        uptr start2, uptr end2) {
  CHECK(start1 <= end1);
  CHECK(start2 <= end2);
  return (end1 < start2) || (end2 < start1);
}

// FIXME: this is thread-unsafe, but should not cause problems most of the time.
// When the shadow is mapped only a single thread usually exists (plus maybe
// several worker threads on Mac, which aren't expected to map big chunks of
// memory).
bool MemoryRangeIsAvailable(uptr range_start, uptr range_end) {
  ProcessMaps procmaps;
  uptr start, end;
  while (procmaps.Next(&start, &end,
                       /*offset*/0, /*filename*/0, /*filename_size*/0)) {
    if (!IntervalsAreSeparate(start, end, range_start, range_end))
      return false;
  }
  return true;
}

void DumpProcessMap() {
  ProcessMaps proc_maps;
  uptr start, end;
  const sptr kBufSize = 4095;
  char *filename = (char*)MmapOrDie(kBufSize, __FUNCTION__);
  Report("Process memory map follows:\n");
  while (proc_maps.Next(&start, &end, /* file_offset */0,
                        filename, kBufSize)) {
    Printf("\t%p-%p\t%s\n", (void*)start, (void*)end, filename);
  }
  Report("End of process memory map.\n");
  UnmapOrDie(filename, kBufSize);
}

const char *GetPwd() {
  return GetEnv("PWD");
}

void DisableCoreDumper() {
  struct rlimit nocore;
  nocore.rlim_cur = 0;
  nocore.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &nocore);
}

void SleepForSeconds(int seconds) {
  sleep(seconds);
}

void SleepForMillis(int millis) {
  usleep(millis * 1000);
}

void Exit(int exitcode) {
  _exit(exitcode);
}

void Abort() {
  abort();
}

int Atexit(void (*function)(void)) {
  return atexit(function);
}

int AtomicInc(int *a) {
#ifdef ANDROID
  return __atomic_inc(a) + 1;
#else
  return __sync_add_and_fetch(a, 1);
#endif
}

u16 AtomicExchange(u16 *a, u16 new_val) {
  return __sync_lock_test_and_set(a, new_val);
}

u8 AtomicExchange(u8 *a, u8 new_val) {
  return __sync_lock_test_and_set(a, new_val);
}

// -------------- sanitizer_libc.h

int internal_sscanf(const char *str, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int res = vsscanf(str, format, args);
  va_end(args);
  return res;
}

}  // namespace __sanitizer

#endif  // __linux__ || __APPLE_
