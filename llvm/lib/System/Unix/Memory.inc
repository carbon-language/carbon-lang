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

/// AllocateRWXMemory - Allocate a slab of memory with read/write/execute
/// permissions.  This is typically used for JIT applications where we want
/// to emit code to the memory then jump to it.  Getting this type of memory
/// is very OS specific.
///
llvm::sys::MemoryBlock 
llvm::sys::Memory::AllocateRWX(unsigned NumBytes, const MemoryBlock* NearBlock,
                               std::string *ErrMsg) {
  if (NumBytes == 0) return MemoryBlock();

  long pageSize = Process::GetPageSize();
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;

  int fd = -1;
#ifdef NEED_DEV_ZERO_FOR_MMAP
  static int zero_fd = open("/dev/zero", O_RDWR);
  if (zero_fd == -1) {
    GetErrno("Can't open /dev/zero device", ErrMsg);
    return MemoryBlock();
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

  void* start = NearBlock ? (unsigned char*)NearBlock->base() + 
                            NearBlock->size() : 0;

  void *pa = ::mmap(start, pageSize*NumPages, PROT_READ|PROT_WRITE|PROT_EXEC,
                    flags, fd, 0);
  if (pa == MAP_FAILED) {
    if (NearBlock) //Try again without a near hint
      return AllocateRWX(NumBytes, 0);

    GetErrno("Can't allocate RWX Memory", ErrMsg);
    return MemoryBlock();
  }
  MemoryBlock result;
  result.Address = pa;
  result.Size = NumPages*pageSize;
  return result;
}

bool llvm::sys::Memory::ReleaseRWX(MemoryBlock &M, std::string *ErrMsg) {
  if (M.Address == 0 || M.Size == 0) return false;
  if (0 != ::munmap(M.Address, M.Size))
    return GetErrno("Can't release RWX Memory", ErrMsg);
  return false;
}

