//===- Cygwin/Memory.cpp - Cygwin Memory Implementation ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Cygwin specific implementation of various Memory
// management utilities
//
//===----------------------------------------------------------------------===//

// Include the generic unix implementation
#include "Unix/Memory.cpp"
#include "llvm/System/Process.h"
#include <sys/types.h>
#include <sys/mman.h>

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Cygwin specific code 
//===          and must not be generic UNIX code (see ../Unix/Memory.cpp)
//===----------------------------------------------------------------------===//

MemoryBlock Memory::AllocateRWX(unsigned NumBytes) {
  if (NumBytes == 0) return MemoryBlock();

  static const long pageSize = Process::GetPageSize();
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;

  void *pa = mmap(0, pageSize*NumPages, PROT_READ|PROT_WRITE|PROT_EXEC,
                  MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (pa == (void*)-1) {
    throw std::string("Can't allocate RWX Memory: ") + strerror(errno);
  }
  MemoryBlock result;
  result.Address = pa;
  result.Size = NumPages*pageSize;
  return result;
}

void Memory::ReleaseRWX(MemoryBlock& M) {
  if (M.Address == 0 || M.Size == 0) return;
  if (0 != munmap(M.Address, M.Size)) {
    throw std::string("Can't release RWX Memory: ") + strerror(errno);
  }
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
