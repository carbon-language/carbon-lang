//===- Win32/Memory.cpp - Win32 Memory Implementation -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of various Memory
// management utilities
//
//===----------------------------------------------------------------------===//

#include <llvm/System/Process.h>
#include "windows.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code.
//===----------------------------------------------------------------------===//

void* Memory::AllocateRWX(Memory&M, unsigned NumBytes) {
  if (NumBytes == 0) return 0;

  unsigned pageSize = Process::GetPageSize();
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;
  void *P = VirtualAlloc(0, NumPages*pageSize, MEM_COMMIT, 
                         PAGE_EXECUTE_READWRITE);
  if (P == 0) {
    throw std::string("Couldn't allocate ") + utostr(NumBytes) + 
        " bytes of executable memory!";
  }
  M.Address = P;
  M.AllocSize = NumBytes;
  return P;
}

void Memory::ReleaseRWX(Memory& M) {
  if (M.Address == 0 ) return;
  VirtualFree(M.Address, M.AllocSize, MEM_DECOMMIT, PAGE_NOACCESS);
}
