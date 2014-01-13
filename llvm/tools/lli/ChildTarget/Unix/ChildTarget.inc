//===- ChildTarget.inc - Child process for external JIT execution for Unix -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Unix-specific parts of the ChildTarget class
// which executes JITed code in a separate process from where it was built.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#if defined(__mips__)
#  if defined(__OpenBSD__)
#    include <mips64/sysarch.h>
#  else
#    include <sys/cachectl.h>
#  endif
#endif

#ifdef __APPLE__
extern "C" void sys_icache_invalidate(const void *Addr, size_t len);
#else
extern "C" void __clear_cache(void *, void*);
#endif

namespace {

struct ConnectionData_t {
  int InputPipe;
  int OutputPipe;

  ConnectionData_t(int in, int out) : InputPipe(in), OutputPipe(out) {}
};

} // namespace

LLIChildTarget::~LLIChildTarget() {
  delete static_cast<ConnectionData_t *>(ConnectionData);
}

// OS-specific methods
void LLIChildTarget::initializeConnection() {
  // Store the parent ends of the pipes
  ConnectionData = (void*)new ConnectionData_t(STDIN_FILENO, STDOUT_FILENO);
}

int LLIChildTarget::WriteBytes(const void *Data, size_t Size) {
  return write(((ConnectionData_t*)ConnectionData)->OutputPipe, Data, Size);
}

int LLIChildTarget::ReadBytes(void *Data, size_t Size) {
  return read(((ConnectionData_t*)ConnectionData)->InputPipe, Data, Size);
}

// The functions below duplicate functionality that is implemented in
// Support/Memory.cpp with the goal of avoiding a dependency on any
// llvm libraries.

uint64_t LLIChildTarget::allocate(uint32_t Alignment, uint32_t Size) {
  if (!Alignment)
    Alignment = 16;

  static const size_t PageSize = getpagesize();
  const size_t NumPages = (Size+PageSize-1)/PageSize;
  Size = NumPages*PageSize;

  int fd = -1;
#ifdef NEED_DEV_ZERO_FOR_MMAP
  static int zero_fd = open("/dev/zero", O_RDWR);
  if (zero_fd == -1)
    return 0;
  fd = zero_fd;
#endif

  int MMFlags = MAP_PRIVATE |
#ifdef HAVE_MMAP_ANONYMOUS
  MAP_ANONYMOUS
#else
  MAP_ANON
#endif
  ; // Ends statement above

  uint64_t Addr = (uint64_t)::mmap(0, Size, PROT_READ | PROT_WRITE, MMFlags, fd, 0);
  if (Addr == (uint64_t)MAP_FAILED)
    return 0;

  // Align the address.
  Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);

  m_AllocatedBufferMap[Addr] = Size;

  // Return aligned address
  return Addr;
}

void LLIChildTarget::makeSectionExecutable(uint64_t Addr, uint32_t Size) {
  // FIXME: We have to mark the memory as RWX because multiple code chunks may
  // be on the same page.  The RemoteTarget interface should be changed to
  // work around that.
  int Result = ::mprotect((void*)Addr, Size, PROT_READ | PROT_WRITE | PROT_EXEC);
  if (Result != 0)
    InvalidateInstructionCache((const void *)Addr, Size);
}

/// InvalidateInstructionCache - Before the JIT can run a block of code
/// that has been emitted it must invalidate the instruction cache on some
/// platforms.
void LLIChildTarget::InvalidateInstructionCache(const void *Addr,
                                        size_t Len) {

// icache invalidation for PPC and ARM.
#if defined(__APPLE__)

#  if (defined(__POWERPC__) || defined (__ppc__) || \
     defined(_POWER) || defined(_ARCH_PPC)) || defined(__arm__)
  sys_icache_invalidate(const_cast<void *>(Addr), Len);
#  endif

#else

#  if (defined(__POWERPC__) || defined (__ppc__) || \
       defined(_POWER) || defined(_ARCH_PPC)) && defined(__GNUC__)
  const size_t LineSize = 32;

  const intptr_t Mask = ~(LineSize - 1);
  const intptr_t StartLine = ((intptr_t) Addr) & Mask;
  const intptr_t EndLine = ((intptr_t) Addr + Len + LineSize - 1) & Mask;

  for (intptr_t Line = StartLine; Line < EndLine; Line += LineSize)
    asm volatile("dcbf 0, %0" : : "r"(Line));
  asm volatile("sync");

  for (intptr_t Line = StartLine; Line < EndLine; Line += LineSize)
    asm volatile("icbi 0, %0" : : "r"(Line));
  asm volatile("isync");
#  elif defined(__arm__) && defined(__GNUC__)
  // FIXME: Can we safely always call this for __GNUC__ everywhere?
  const char *Start = static_cast<const char *>(Addr);
  const char *End = Start + Len;
  __clear_cache(const_cast<char *>(Start), const_cast<char *>(End));
#  elif defined(__mips__)
  const char *Start = static_cast<const char *>(Addr);
  cacheflush(const_cast<char *>(Start), Len, BCACHE);
#  endif

#endif  // end apple
}

void LLIChildTarget::releaseMemory(uint64_t Addr, uint32_t Size) {
  ::munmap((void*)Addr, Size);
}
