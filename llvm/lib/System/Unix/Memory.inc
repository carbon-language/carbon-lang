//===- Unix/Memory.cpp - Generic UNIX System Configuration ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some functions for various memory management utilities.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/System/Process.h"

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

namespace llvm {

/// AllocateRWXMemory - Allocate a slab of memory with read/write/execute
/// permissions.  This is typically used for JIT applications where we want
/// to emit code to the memory then jump to it.  Getting this type of memory
/// is very OS specific.
///
MemoryBlock Memory::AllocateRWX(unsigned NumBytes, const MemoryBlock* NearBlock) {
  if (NumBytes == 0) return MemoryBlock();

  long pageSize = Process::GetPageSize();
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;

  int fd = -1;
#ifdef NEED_DEV_ZERO_FOR_MMAP
  static int zero_fd = open("/dev/zero", O_RDWR);
  if (zero_fd == -1) {
    ThrowErrno("Can't open /dev/zero device");
  }
  fd = zero_fd;
#endif

  int flags = MAP_PRIVATE |
#ifdef HAVE_MMAP_ANONYMOUS
  MAP_ANONYMOUS
#else
  MAP_ANON
#endif
  ;

  void* start = NearBlock ? (unsigned char*) NearBlock->base() + NearBlock->size() : 0;

  void *pa = ::mmap(start, pageSize*NumPages, PROT_READ|PROT_WRITE|PROT_EXEC,
                    flags, fd, 0);
  if (pa == MAP_FAILED) {
    if (NearBlock) //Try again without a near hint
      return AllocateRWX(NumBytes, 0);
    else
      ThrowErrno("Can't allocate RWX Memory");
  }
  MemoryBlock result;
  result.Address = pa;
  result.Size = NumPages*pageSize;
  return result;
}

void Memory::ReleaseRWX(MemoryBlock& M) {
  if (M.Address == 0 || M.Size == 0) return;
  if (0 != ::munmap(M.Address, M.Size)) {
    ThrowErrno("Can't release RWX Memory");
  }
}

}

