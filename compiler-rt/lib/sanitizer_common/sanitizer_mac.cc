//===-- sanitizer_mac.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries and
// implements OSX-specific functions.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_MAC

// Use 64-bit inodes in file operations. ASan does not support OS X 10.5, so
// the clients will most certainly use 64-bit ones as well.
#ifndef _DARWIN_USE_64_BIT_INODE
#define _DARWIN_USE_64_BIT_INODE 1
#endif
#include <stdio.h>

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_mac.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"

#include <crt_externs.h>  // for _NSGetEnviron
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#include <libkern/OSAtomic.h>
#include <errno.h>

namespace __sanitizer {

#include "sanitizer_syscall_generic.inc"

// ---------------------- sanitizer_libc.h
uptr internal_mmap(void *addr, size_t length, int prot, int flags,
                   int fd, u64 offset) {
  return (uptr)mmap(addr, length, prot, flags, fd, offset);
}

uptr internal_munmap(void *addr, uptr length) {
  return munmap(addr, length);
}

uptr internal_close(fd_t fd) {
  return close(fd);
}

uptr internal_open(const char *filename, int flags) {
  return open(filename, flags);
}

uptr internal_open(const char *filename, int flags, u32 mode) {
  return open(filename, flags, mode);
}

uptr OpenFile(const char *filename, bool write) {
  return internal_open(filename,
      write ? O_WRONLY | O_CREAT : O_RDONLY, 0660);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  return read(fd, buf, count);
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  return write(fd, buf, count);
}

uptr internal_stat(const char *path, void *buf) {
  return stat(path, (struct stat *)buf);
}

uptr internal_lstat(const char *path, void *buf) {
  return lstat(path, (struct stat *)buf);
}

uptr internal_fstat(fd_t fd, void *buf) {
  return fstat(fd, (struct stat *)buf);
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup2(int oldfd, int newfd) {
  return dup2(oldfd, newfd);
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
  return readlink(path, buf, bufsize);
}

uptr internal_sched_yield() {
  return sched_yield();
}

void internal__exit(int exitcode) {
  _exit(exitcode);
}

uptr internal_getpid() {
  return getpid();
}

int internal_sigaction(int signum, const void *act, void *oldact) {
  return sigaction(signum,
                   (struct sigaction *)act, (struct sigaction *)oldact);
}

// ----------------- sanitizer_common.h
bool FileExists(const char *filename) {
  struct stat st;
  if (stat(filename, &st))
    return false;
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
}

uptr GetTid() {
  return reinterpret_cast<uptr>(pthread_self());
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  uptr stacksize = pthread_get_stacksize_np(pthread_self());
  // pthread_get_stacksize_np() returns an incorrect stack size for the main
  // thread on Mavericks. See
  // https://code.google.com/p/address-sanitizer/issues/detail?id=261
  if ((GetMacosVersion() == MACOS_VERSION_MAVERICKS) && at_initialization &&
      stacksize == (1 << 19))  {
    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);
    // Most often rl.rlim_cur will be the desired 8M.
    if (rl.rlim_cur < kMaxThreadStackSize) {
      stacksize = rl.rlim_cur;
    } else {
      stacksize = kMaxThreadStackSize;
    }
  }
  void *stackaddr = pthread_get_stackaddr_np(pthread_self());
  *stack_top = (uptr)stackaddr;
  *stack_bottom = *stack_top - stacksize;
}

const char *GetEnv(const char *name) {
  char ***env_ptr = _NSGetEnviron();
  if (!env_ptr) {
    Report("_NSGetEnviron() returned NULL. Please make sure __asan_init() is "
           "called after libSystem_initializer().\n");
    CHECK(env_ptr);
  }
  char **environ = *env_ptr;
  CHECK(environ);
  uptr name_len = internal_strlen(name);
  while (*environ != 0) {
    uptr len = internal_strlen(*environ);
    if (len > name_len) {
      const char *p = *environ;
      if (!internal_memcmp(p, name, name_len) &&
          p[name_len] == '=') {  // Match.
        return *environ + name_len + 1;  // String starting after =.
      }
    }
    environ++;
  }
  return 0;
}

void ReExec() {
  UNIMPLEMENTED();
}

void PrepareForSandboxing() {
  // Nothing here for now.
}

uptr GetPageSize() {
  return sysconf(_SC_PAGESIZE);
}

BlockingMutex::BlockingMutex(LinkerInitialized) {
  // We assume that OS_SPINLOCK_INIT is zero
}

BlockingMutex::BlockingMutex() {
  internal_memset(this, 0, sizeof(*this));
}

void BlockingMutex::Lock() {
  CHECK(sizeof(OSSpinLock) <= sizeof(opaque_storage_));
  CHECK_EQ(OS_SPINLOCK_INIT, 0);
  CHECK_NE(owner_, (uptr)pthread_self());
  OSSpinLockLock((OSSpinLock*)&opaque_storage_);
  CHECK(!owner_);
  owner_ = (uptr)pthread_self();
}

void BlockingMutex::Unlock() {
  CHECK(owner_ == (uptr)pthread_self());
  owner_ = 0;
  OSSpinLockUnlock((OSSpinLock*)&opaque_storage_);
}

void BlockingMutex::CheckLocked() {
  CHECK_EQ((uptr)pthread_self(), owner_);
}

u64 NanoTime() {
  return 0;
}

uptr GetTlsSize() {
  return 0;
}

void InitTlsSize() {
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#ifndef SANITIZER_GO
  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_addr = stack_bottom;
  *stk_size = stack_top - stack_bottom;
  *tls_addr = 0;
  *tls_size = 0;
#else
  *stk_addr = 0;
  *stk_size = 0;
  *tls_addr = 0;
  *tls_size = 0;
#endif
}

uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  MemoryMappingLayout memory_mapping(false);
  return memory_mapping.DumpListOfModules(modules, max_modules, filter);
}

bool IsDeadlySignal(int signum) {
  return (signum == SIGSEGV || signum == SIGBUS) && common_flags()->handle_segv;
}

MacosVersion cached_macos_version = MACOS_VERSION_UNINITIALIZED;

MacosVersion GetMacosVersionInternal() {
  int mib[2] = { CTL_KERN, KERN_OSRELEASE };
  char version[100];
  uptr len = 0, maxlen = sizeof(version) / sizeof(version[0]);
  for (uptr i = 0; i < maxlen; i++) version[i] = '\0';
  // Get the version length.
  CHECK_NE(sysctl(mib, 2, 0, &len, 0, 0), -1);
  CHECK_LT(len, maxlen);
  CHECK_NE(sysctl(mib, 2, version, &len, 0, 0), -1);
  switch (version[0]) {
    case '9': return MACOS_VERSION_LEOPARD;
    case '1': {
      switch (version[1]) {
        case '0': return MACOS_VERSION_SNOW_LEOPARD;
        case '1': return MACOS_VERSION_LION;
        case '2': return MACOS_VERSION_MOUNTAIN_LION;
        case '3': return MACOS_VERSION_MAVERICKS;
        default: return MACOS_VERSION_UNKNOWN;
      }
    }
    default: return MACOS_VERSION_UNKNOWN;
  }
}

MacosVersion GetMacosVersion() {
  atomic_uint32_t *cache =
      reinterpret_cast<atomic_uint32_t*>(&cached_macos_version);
  MacosVersion result =
      static_cast<MacosVersion>(atomic_load(cache, memory_order_acquire));
  if (result == MACOS_VERSION_UNINITIALIZED) {
    result = GetMacosVersionInternal();
    atomic_store(cache, result, memory_order_release);
  }
  return result;
}

}  // namespace __sanitizer

#endif  // SANITIZER_MAC
