//===- Unix/DynamicLibrary.cpp - Generic UNIX Dynamic Library ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the generic UNIX variant of DynamicLibrary
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "sys/stat.h"

namespace llvm {
using namespace sys;

DynamicLibrary::DynamicLibrary() : handle(0) {
#if defined (HAVE_DLOPEN)
  if ((handle = dlopen(0, RTLD_NOW | RTLD_GLOBAL)) == 0)
    throw std::string( dlerror() );
#else
  assert(!"Dynamic object linking not implemented for this platform");
#endif
}

DynamicLibrary::DynamicLibrary(const char *filename) : handle(0) {
#if defined (HAVE_DLOPEN)
  if ((handle = dlopen (filename, RTLD_NOW | RTLD_GLOBAL)) == 0)
    throw std::string( dlerror() );
#else
  assert (!"Dynamic object linking not implemented for this platform");
#endif
}

DynamicLibrary::~DynamicLibrary() {
  assert(handle != 0 && "Invalid DynamicLibrary handle");
#if defined (HAVE_DLOPEN)
  dlclose(handle);
#else
  assert (!"Dynamic object linking not implemented for this platform");
#endif
}

void *DynamicLibrary::GetAddressOfSymbol(const char *symbolName) {
  assert(handle != 0 && "Invalid DynamicLibrary handle");
#if defined(HAVE_DLOPEN)
    return dlsym (handle, symbolName);
#else
  assert (0 && "Dynamic symbol lookup not implemented for this platform");
#endif
}

}
