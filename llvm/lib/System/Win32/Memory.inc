//===- Win32/Memory.cpp - Win32 Memory Implementation -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of various Memory
// management utilities
//
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include "llvm/System/Process.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

MemoryBlock Memory::AllocateRWX(unsigned NumBytes) {
  if (NumBytes == 0) return MemoryBlock();

  static const long pageSize = Process::GetPageSize();
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;

  void *pa = VirtualAlloc(NULL, NumPages*pageSize, MEM_COMMIT,
                  PAGE_EXECUTE_READWRITE);
  if (pa == NULL) {
    ThrowError("Can't allocate RWX Memory: ");
  }

  MemoryBlock result;
  result.Address = pa;
  result.Size = NumPages*pageSize;
  return result;
}

void Memory::ReleaseRWX(MemoryBlock& M) {
  if (M.Address == 0 || M.Size == 0) return;
  if (!VirtualFree(M.Address, 0, MEM_RELEASE)) {
    ThrowError("Can't release RWX Memory: ");
  }
}

}

