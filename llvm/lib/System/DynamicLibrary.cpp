//===-- DynamicLibrary.cpp - Runtime link/load libraries --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system DynamicLibrary concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/System/DynamicLibrary.h"
#include "llvm/Config/dlfcn.h"
#include <cassert>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code. 
//===----------------------------------------------------------------------===//

#ifdef HAVE_LTDL_H

namespace llvm {

using namespace sys;

#ifdef HAVE_LT_DLOPEN

DynamicLibrary::DynamicLibrary() : handle(0) {
  if (0 != lt_dlinit())
    throw std::string(lt_dlerror());

  handle = lt_dlopen(0);

  if (handle == 0)
    throw std::string("Can't open program as dynamic library");
}

DynamicLibrary::DynamicLibrary(const char*filename) : handle(0) {
  if (0 != lt_dlinit())
    throw std::string(lt_dlerror());

  handle = lt_dlopen(filename);

  if (handle == 0)
    handle = lt_dlopenext(filename);

  if (handle == 0)
    throw std::string("Can't open dynamic library:") + filename;
}

DynamicLibrary::~DynamicLibrary() {
  if (handle)
    lt_dlclose((lt_dlhandle)handle);

  lt_dlexit();
}

void *DynamicLibrary::GetAddressOfSymbol(const char *symbolName) {
  assert(handle != 0 && "Invalid DynamicLibrary handle");
  return lt_dlsym((lt_dlhandle) handle,symbolName);
}

#else
DynamicLibrary::DynamicLibrary(const char*filename) : handle(0) {
  assert(!"Have ltdl.h but not libltdl.a!");
}

DynamicLibrary::~DynamicLibrary() {
  assert(!"Have ltdl.h but not libltdl.a!");
}

void *DynamicLibrary::GetAddressOfSymbol(const char *symbolName) {
  assert(!"Have ltdl.h but not libltdl.a!");
  return 0;
}

#endif // HAVE_DLOPEN

} // namespace llvm

#else // HAVE_LTDL_H

#include "platform/DynamicLibrary.cpp"

#endif
