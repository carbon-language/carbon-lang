//===- Win32/DynamicLibrary.cpp - Win32 DL Implementation -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the DynamicLibrary
//
//===----------------------------------------------------------------------===//

#include "Win32.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

DynamicLibrary::DynamicLibrary() : handle(0) {
  handle = (void*) GetModuleHandle(NULL);
  
  if (handle == 0) {
    ThrowError("Can't GetModuleHandle: ");
  }
}

DynamicLibrary::DynamicLibrary(const char*filename) : handle(0) {
  handle = LoadLibrary(filename);

  if (handle == 0) {
    ThrowError("Can't LoadLibrary: ");
  }
}

DynamicLibrary::~DynamicLibrary() {
  assert(handle !=0 && "Invalid DynamicLibrary handle");
  if (handle)
    FreeLibrary((HMODULE*)handle);
}

void *DynamicLibrary::GetAddressOfSymbol(const char *symbolName) {
  assert(handle !=0 && "Invalid DynamicLibrary handle");
  return (void*) GetProcAddress((HMODULE*)handle, symbolName);
}

}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
