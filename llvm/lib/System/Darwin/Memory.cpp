//===- Darwin/Memory.cpp - Darwin Memory Implementation ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Darwin specific implementation of various Memory
// management utilities
//
//===----------------------------------------------------------------------===//

// Include the generic unix implementation
#include "../Unix/Memory.cpp"
#include "llvm/System/Process.h"
#include <sys/mman.h>

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Darwin-specific code 
//===          and must not be generic UNIX code (see ../Unix/Memory.cpp)
//===----------------------------------------------------------------------===//

/// AllocateRWXMemory - Allocate a slab of memory with read/write/execute
/// permissions.  This is typically used for JIT applications where we want
/// to emit code to the memory and then jump to it.  Getting this type of memory
/// is very OS-specific.
///
MemoryBlock Memory::AllocateRWX(unsigned NumBytes) {
  if (NumBytes == 0) return MemoryBlock();

  static const long pageSize = Process::GetPageSize();
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;

  void *pa = mmap(0, pageSize*NumPages, PROT_READ|PROT_WRITE|PROT_EXEC,
                  MAP_PRIVATE|MAP_ANON, -1, 0);
  if (pa == MAP_FAILED) {
    char msg[MAXPATHLEN];
    strerror_r(errno, msg, MAXPATHLEN-1);
    throw std::string("Can't allocate RWX Memory: ") + msg;
  }
  MemoryBlock result;
  result.Address = pa;
  result.Size = NumPages*pageSize;
  return result;
}

void Memory::ReleaseRWX(MemoryBlock& M) {
  if (M.Address == 0 || M.Size == 0) return;
  if (0 != munmap(M.Address, M.Size)) {
    char msg[MAXPATHLEN];
    strerror_r(errno, msg, MAXPATHLEN-1);
    throw std::string("Can't release RWX Memory: ") + msg;
  }
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
