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

#include "sanitizer_platform.h"
#if SANITIZER_LINUX || SANITIZER_MAC

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"

#include <sys/mman.h>

namespace __sanitizer {

// ------------- sanitizer_common.h
uptr GetMmapGranularity() {
  return GetPageSize();
}

uptr GetMaxVirtualAddress() {
#if SANITIZER_WORDSIZE == 64
# if defined(__powerpc64__)
  // On PowerPC64 we have two different address space layouts: 44- and 46-bit.
  // We somehow need to figure our which one we are using now and choose
  // one of 0x00000fffffffffffUL and 0x00003fffffffffffUL.
  // Note that with 'ulimit -s unlimited' the stack is moved away from the top
  // of the address space, so simply checking the stack address is not enough.
  return (1ULL << 44) - 1;  // 0x00000fffffffffffUL
# elif defined(__aarch64__)
  return (1ULL << 39) - 1;
# else
  return (1ULL << 47) - 1;  // 0x00007fffffffffffUL;
# endif
#else  // SANITIZER_WORDSIZE == 32
  // FIXME: We can probably lower this on Android?
  return (1ULL << 32) - 1;  // 0xffffffff;
#endif  // SANITIZER_WORDSIZE
}

void *MmapOrDie(uptr size, const char *mem_type) {
  size = RoundUpTo(size, GetPageSizeCached());
  uptr res = internal_mmap(0, size,
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANON, -1, 0);
  int reserrno;
  if (internal_iserror(res, &reserrno)) {
    static int recursion_count;
    if (recursion_count) {
      // The Report() and CHECK calls below may call mmap recursively and fail.
      // If we went into recursion, just die.
      RawWrite("ERROR: Failed to mmap\n");
      Die();
    }
    recursion_count++;
    Report("ERROR: %s failed to allocate 0x%zx (%zd) bytes of %s: %d\n",
           SanitizerToolName, size, size, mem_type, reserrno);
    DumpProcessMap();
    CHECK("unable to mmap" && 0);
  }
  return (void *)res;
}

void UnmapOrDie(void *addr, uptr size) {
  if (!addr || !size) return;
  uptr res = internal_munmap(addr, size);
  if (internal_iserror(res)) {
    Report("ERROR: %s failed to deallocate 0x%zx (%zd) bytes at address %p\n",
           SanitizerToolName, size, size, addr);
    CHECK("unable to unmap" && 0);
  }
}

void *MmapNoReserveOrDie(uptr size, const char *mem_type) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap(0,
      RoundUpTo(size, PageSize),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,
      -1, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno)) {
    Report("ERROR: "
           "%s failed to allocate noreserve 0x%zx (%zd) bytes for '%s' (%d)\n",
           SanitizerToolName, size, size, mem_type, reserrno);
    CHECK("unable to mmap" && 0);
  }
  return (void *)p;
}

void *MmapFixedNoReserve(uptr fixed_addr, uptr size) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap((void*)(fixed_addr & ~(PageSize - 1)),
      RoundUpTo(size, PageSize),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
      -1, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno))
    Report("ERROR: "
           "%s failed to allocate 0x%zx (%zd) bytes at address %zu (%d)\n",
           SanitizerToolName, size, size, fixed_addr, reserrno);
  return (void *)p;
}

void *MmapFixedOrDie(uptr fixed_addr, uptr size) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap((void*)(fixed_addr & ~(PageSize - 1)),
      RoundUpTo(size, PageSize),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_FIXED,
      -1, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno)) {
    Report("ERROR:"
           " %s failed to allocate 0x%zx (%zd) bytes at address %zu (%d)\n",
           SanitizerToolName, size, size, fixed_addr, reserrno);
    CHECK("unable to mmap" && 0);
  }
  return (void *)p;
}

void *Mprotect(uptr fixed_addr, uptr size) {
  return (void *)internal_mmap((void*)fixed_addr, size,
                               PROT_NONE,
                               MAP_PRIVATE | MAP_ANON | MAP_FIXED |
                               MAP_NORESERVE, -1, 0);
}

void *MapFileToMemory(const char *file_name, uptr *buff_size) {
  uptr openrv = OpenFile(file_name, false);
  CHECK(!internal_iserror(openrv));
  fd_t fd = openrv;
  uptr fsize = internal_filesize(fd);
  CHECK_NE(fsize, (uptr)-1);
  CHECK_GT(fsize, 0);
  *buff_size = RoundUpTo(fsize, GetPageSizeCached());
  uptr map = internal_mmap(0, *buff_size, PROT_READ, MAP_PRIVATE, fd, 0);
  return internal_iserror(map) ? 0 : (void *)map;
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
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr start, end;
  while (proc_maps.Next(&start, &end,
                        /*offset*/0, /*filename*/0, /*filename_size*/0,
                        /*protection*/0)) {
    if (!IntervalsAreSeparate(start, end, range_start, range_end))
      return false;
  }
  return true;
}

void DumpProcessMap() {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr start, end;
  const sptr kBufSize = 4095;
  char *filename = (char*)MmapOrDie(kBufSize, __func__);
  Report("Process memory map follows:\n");
  while (proc_maps.Next(&start, &end, /* file_offset */0,
                        filename, kBufSize, /* protection */0)) {
    Printf("\t%p-%p\t%s\n", (void*)start, (void*)end, filename);
  }
  Report("End of process memory map.\n");
  UnmapOrDie(filename, kBufSize);
}

const char *GetPwd() {
  return GetEnv("PWD");
}

char *FindPathToBinary(const char *name) {
  const char *path = GetEnv("PATH");
  if (!path)
    return 0;
  uptr name_len = internal_strlen(name);
  InternalScopedBuffer<char> buffer(kMaxPathLength);
  const char *beg = path;
  while (true) {
    const char *end = internal_strchrnul(beg, ':');
    uptr prefix_len = end - beg;
    if (prefix_len + name_len + 2 <= kMaxPathLength) {
      internal_memcpy(buffer.data(), beg, prefix_len);
      buffer[prefix_len] = '/';
      internal_memcpy(&buffer[prefix_len + 1], name, name_len);
      buffer[prefix_len + 1 + name_len] = '\0';
      if (FileExists(buffer.data()))
        return internal_strdup(buffer.data());
    }
    if (*end == '\0') break;
    beg = end + 1;
  }
  return 0;
}

void MaybeOpenReportFile() {
  if (!log_to_file) return;
  uptr pid = internal_getpid();
  // If in tracer, use the parent's file.
  if (pid == stoptheworld_tracer_pid)
    pid = stoptheworld_tracer_ppid;
  if (report_fd_pid == pid) return;
  InternalScopedBuffer<char> report_path_full(4096);
  internal_snprintf(report_path_full.data(), report_path_full.size(),
                    "%s.%zu", report_path_prefix, pid);
  uptr openrv = OpenFile(report_path_full.data(), true);
  if (internal_iserror(openrv)) {
    report_fd = kStderrFd;
    log_to_file = false;
    Report("ERROR: Can't open file: %s\n", report_path_full.data());
    Die();
  }
  if (report_fd != kInvalidFd) {
    // We're in the child. Close the parent's log.
    internal_close(report_fd);
  }
  report_fd = openrv;
  report_fd_pid = pid;
}

void RawWrite(const char *buffer) {
  static const char *kRawWriteError =
      "RawWrite can't output requested buffer!\n";
  uptr length = (uptr)internal_strlen(buffer);
  MaybeOpenReportFile();
  if (length != internal_write(report_fd, buffer, length)) {
    internal_write(report_fd, kRawWriteError, internal_strlen(kRawWriteError));
    Die();
  }
}

bool GetCodeRangeForFile(const char *module, uptr *start, uptr *end) {
  uptr s, e, off, prot;
  InternalScopedString buff(4096);
  MemoryMappingLayout proc_maps(/*cache_enabled*/false);
  while (proc_maps.Next(&s, &e, &off, buff.data(), buff.size(), &prot)) {
    if ((prot & MemoryMappingLayout::kProtectionExecute) != 0
        && internal_strcmp(module, buff.data()) == 0) {
      *start = s;
      *end = e;
      return true;
    }
  }
  return false;
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX || SANITIZER_MAC
