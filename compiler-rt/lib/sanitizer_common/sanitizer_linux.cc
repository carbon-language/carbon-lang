//===-- sanitizer_linux.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//
#ifdef __linux__

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <sys/prctl.h>

// Are we using 32-bit or 64-bit syscalls?
// x32 (which defines __x86_64__) has SANITIZER_WORDSIZE == 32
// but it still needs to use 64-bit syscalls.
#if defined(__x86_64__) || SANITIZER_WORDSIZE == 64
# define SANITIZER_LINUX_USES_64BIT_SYSCALLS 1
#else
# define SANITIZER_LINUX_USES_64BIT_SYSCALLS 0
#endif

namespace __sanitizer {

// --------------- sanitizer_libc.h
void *internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return (void *)syscall(__NR_mmap, addr, length, prot, flags, fd, offset);
#else
  return (void *)syscall(__NR_mmap2, addr, length, prot, flags, fd, offset);
#endif
}

int internal_munmap(void *addr, uptr length) {
  return syscall(__NR_munmap, addr, length);
}

int internal_close(fd_t fd) {
  return syscall(__NR_close, fd);
}

fd_t internal_open(const char *filename, bool write) {
  return syscall(__NR_open, filename,
      write ? O_WRONLY | O_CREAT /*| O_CLOEXEC*/ : O_RDONLY, 0660);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  sptr res;
  HANDLE_EINTR(res, (sptr)syscall(__NR_read, fd, buf, count));
  return res;
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  sptr res;
  HANDLE_EINTR(res, (sptr)syscall(__NR_write, fd, buf, count));
  return res;
}

uptr internal_filesize(fd_t fd) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  struct stat st;
  if (syscall(__NR_fstat, fd, &st))
    return -1;
#else
  struct stat64 st;
  if (syscall(__NR_fstat64, fd, &st))
    return -1;
#endif
  return (uptr)st.st_size;
}

int internal_dup2(int oldfd, int newfd) {
  return syscall(__NR_dup2, oldfd, newfd);
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
  return (uptr)syscall(__NR_readlink, path, buf, bufsize);
}

int internal_sched_yield() {
  return syscall(__NR_sched_yield);
}

// ----------------- sanitizer_common.h
bool FileExists(const char *filename) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  struct stat st;
  if (syscall(__NR_stat, filename, &st))
    return false;
#else
  struct stat64 st;
  if (syscall(__NR_stat64, filename, &st))
    return false;
#endif
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
}

uptr GetTid() {
  return syscall(__NR_gettid);
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  static const uptr kMaxThreadStackSize = 256 * (1 << 20);  // 256M
  CHECK(stack_top);
  CHECK(stack_bottom);
  if (at_initialization) {
    // This is the main thread. Libpthread may not be initialized yet.
    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);

    // Find the mapping that contains a stack variable.
    MemoryMappingLayout proc_maps;
    uptr start, end, offset;
    uptr prev_end = 0;
    while (proc_maps.Next(&start, &end, &offset, 0, 0)) {
      if ((uptr)&rl < end)
        break;
      prev_end = end;
    }
    CHECK((uptr)&rl >= start && (uptr)&rl < end);

    // Get stacksize from rlimit, but clip it so that it does not overlap
    // with other mappings.
    uptr stacksize = rl.rlim_cur;
    if (stacksize > end - prev_end)
      stacksize = end - prev_end;
    // When running with unlimited stack size, we still want to set some limit.
    // The unlimited stack size is caused by 'ulimit -s unlimited'.
    // Also, for some reason, GNU make spawns subprocesses with unlimited stack.
    if (stacksize > kMaxThreadStackSize)
      stacksize = kMaxThreadStackSize;
    *stack_top = end;
    *stack_bottom = end - stacksize;
    return;
  }
  pthread_attr_t attr;
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  uptr stacksize = 0;
  void *stackaddr = 0;
  pthread_attr_getstack(&attr, &stackaddr, (size_t*)&stacksize);
  pthread_attr_destroy(&attr);

  *stack_top = (uptr)stackaddr + stacksize;
  *stack_bottom = (uptr)stackaddr;
  CHECK(stacksize < kMaxThreadStackSize);  // Sanity check.
}

// Like getenv, but reads env directly from /proc and does not use libc.
// This function should be called first inside __asan_init.
const char *GetEnv(const char *name) {
  static char *environ;
  static uptr len;
  static bool inited;
  if (!inited) {
    inited = true;
    uptr environ_size;
    len = ReadFileToBuffer("/proc/self/environ",
                           &environ, &environ_size, 1 << 26);
  }
  if (!environ || len == 0) return 0;
  uptr namelen = internal_strlen(name);
  const char *p = environ;
  while (*p != '\0') {  // will happen at the \0\0 that terminates the buffer
    // proc file has the format NAME=value\0NAME=value\0NAME=value\0...
    const char* endp =
        (char*)internal_memchr(p, '\0', len - (p - environ));
    if (endp == 0)  // this entry isn't NUL terminated
      return 0;
    else if (!internal_memcmp(p, name, namelen) && p[namelen] == '=')  // Match.
      return p + namelen + 1;  // point after =
    p = endp + 1;
  }
  return 0;  // Not found.
}

void ReExec() {
  static const int kMaxArgv = 100;
  InternalScopedBuffer<char*> argv(kMaxArgv + 1);
  static char *buff;
  uptr buff_size = 0;
  ReadFileToBuffer("/proc/self/cmdline", &buff, &buff_size, 1024 * 1024);
  argv[0] = buff;
  int argc, i;
  for (argc = 1, i = 1; ; i++) {
    if (buff[i] == 0) {
      if (buff[i+1] == 0) break;
      argv[argc] = &buff[i+1];
      CHECK_LE(argc, kMaxArgv);  // FIXME: make this more flexible.
      argc++;
    }
  }
  argv[argc] = 0;
  execv(argv[0], argv.data());
}

void PrepareForSandboxing() {
  // Some kinds of sandboxes may forbid filesystem access, so we won't be able
  // to read the file mappings from /proc/self/maps. Luckily, neither the
  // process will be able to load additional libraries, so it's fine to use the
  // cached mappings.
  MemoryMappingLayout::CacheMemoryMappings();
}

// ----------------- sanitizer_procmaps.h
// Linker initialized.
ProcSelfMapsBuff MemoryMappingLayout::cached_proc_self_maps_;
StaticSpinMutex MemoryMappingLayout::cache_lock_;  // Linker initialized.

MemoryMappingLayout::MemoryMappingLayout() {
  proc_self_maps_.len =
      ReadFileToBuffer("/proc/self/maps", &proc_self_maps_.data,
                       &proc_self_maps_.mmaped_size, 1 << 26);
  if (proc_self_maps_.mmaped_size == 0) {
    LoadFromCache();
    CHECK_GT(proc_self_maps_.len, 0);
  }
  // internal_write(2, proc_self_maps_.data, proc_self_maps_.len);
  Reset();
  // FIXME: in the future we may want to cache the mappings on demand only.
  CacheMemoryMappings();
}

MemoryMappingLayout::~MemoryMappingLayout() {
  // Only unmap the buffer if it is different from the cached one. Otherwise
  // it will be unmapped when the cache is refreshed.
  if (proc_self_maps_.data != cached_proc_self_maps_.data) {
    UnmapOrDie(proc_self_maps_.data, proc_self_maps_.mmaped_size);
  }
}

void MemoryMappingLayout::Reset() {
  current_ = proc_self_maps_.data;
}

// static
void MemoryMappingLayout::CacheMemoryMappings() {
  SpinMutexLock l(&cache_lock_);
  // Don't invalidate the cache if the mappings are unavailable.
  ProcSelfMapsBuff old_proc_self_maps;
  old_proc_self_maps = cached_proc_self_maps_;
  cached_proc_self_maps_.len =
      ReadFileToBuffer("/proc/self/maps", &cached_proc_self_maps_.data,
                       &cached_proc_self_maps_.mmaped_size, 1 << 26);
  if (cached_proc_self_maps_.mmaped_size == 0) {
    cached_proc_self_maps_ = old_proc_self_maps;
  } else {
    if (old_proc_self_maps.mmaped_size) {
      UnmapOrDie(old_proc_self_maps.data,
                 old_proc_self_maps.mmaped_size);
    }
  }
}

void MemoryMappingLayout::LoadFromCache() {
  SpinMutexLock l(&cache_lock_);
  if (cached_proc_self_maps_.data) {
    proc_self_maps_ = cached_proc_self_maps_;
  }
}

// Parse a hex value in str and update str.
static uptr ParseHex(char **str) {
  uptr x = 0;
  char *s;
  for (s = *str; ; s++) {
    char c = *s;
    uptr v = 0;
    if (c >= '0' && c <= '9')
      v = c - '0';
    else if (c >= 'a' && c <= 'f')
      v = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      v = c - 'A' + 10;
    else
      break;
    x = x * 16 + v;
  }
  *str = s;
  return x;
}

static bool IsOnOf(char c, char c1, char c2) {
  return c == c1 || c == c2;
}

static bool IsDecimal(char c) {
  return c >= '0' && c <= '9';
}

bool MemoryMappingLayout::Next(uptr *start, uptr *end, uptr *offset,
                               char filename[], uptr filename_size) {
  char *last = proc_self_maps_.data + proc_self_maps_.len;
  if (current_ >= last) return false;
  uptr dummy;
  if (!start) start = &dummy;
  if (!end) end = &dummy;
  if (!offset) offset = &dummy;
  char *next_line = (char*)internal_memchr(current_, '\n', last - current_);
  if (next_line == 0)
    next_line = last;
  // Example: 08048000-08056000 r-xp 00000000 03:0c 64593   /foo/bar
  *start = ParseHex(&current_);
  CHECK_EQ(*current_++, '-');
  *end = ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  CHECK(IsOnOf(*current_++, '-', 'r'));
  CHECK(IsOnOf(*current_++, '-', 'w'));
  CHECK(IsOnOf(*current_++, '-', 'x'));
  CHECK(IsOnOf(*current_++, 's', 'p'));
  CHECK_EQ(*current_++, ' ');
  *offset = ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  ParseHex(&current_);
  CHECK_EQ(*current_++, ':');
  ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  while (IsDecimal(*current_))
    current_++;
  CHECK_EQ(*current_++, ' ');
  // Skip spaces.
  while (current_ < next_line && *current_ == ' ')
    current_++;
  // Fill in the filename.
  uptr i = 0;
  while (current_ < next_line) {
    if (filename && i < filename_size - 1)
      filename[i++] = *current_;
    current_++;
  }
  if (filename && i < filename_size)
    filename[i] = 0;
  current_ = next_line + 1;
  return true;
}

// Gets the object name and the offset by walking MemoryMappingLayout.
bool MemoryMappingLayout::GetObjectNameAndOffset(uptr addr, uptr *offset,
                                                 char filename[],
                                                 uptr filename_size) {
  return IterateForObjectNameAndOffset(addr, offset, filename, filename_size);
}

bool SanitizerSetThreadName(const char *name) {
  return 0 == prctl(PR_SET_NAME, (unsigned long)name, 0, 0, 0);  // NOLINT
}

bool SanitizerGetThreadName(char *name, int max_len) {
  char buff[17];
  if (prctl(PR_GET_NAME, (unsigned long)buff, 0, 0, 0))  // NOLINT
    return false;
  internal_strncpy(name, buff, max_len);
  name[max_len] = 0;
  return true;
}

}  // namespace __sanitizer

#endif  // __linux__
