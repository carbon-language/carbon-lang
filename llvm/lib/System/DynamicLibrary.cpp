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
#include "ltdl.h"
#include <cassert>

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code. 
//===----------------------------------------------------------------------===//

static bool did_initialize_ltdl = false;

static inline void check_ltdl_initialization() {
  if (!did_initialize_ltdl) {
    if (0 != lt_dlinit())
      throw std::string(lt_dlerror());
    did_initialize_ltdl = true;
  }
}

static std::vector<lt_dlhandle> OpenedHandles;

namespace llvm {

using namespace sys;

DynamicLibrary::DynamicLibrary() : handle(0) {
  check_ltdl_initialization();

  lt_dlhandle a_handle = lt_dlopen(0);

  if (a_handle == 0)
    throw std::string("Can't open program as dynamic library");
  
  handle = a_handle;
  OpenedHandles.push_back(a_handle);
}

DynamicLibrary::DynamicLibrary(const char*filename) : handle(0) {
  check_ltdl_initialization();

  lt_dlhandle a_handle = lt_dlopen(filename);

  if (a_handle == 0)
    a_handle = lt_dlopenext(filename);

  if (a_handle == 0)
    throw std::string("Can't open :") + filename + ": " + lt_dlerror();

  handle = a_handle;
  OpenedHandles.push_back(a_handle);
}

DynamicLibrary::~DynamicLibrary() {
  lt_dlhandle a_handle = (lt_dlhandle) handle;
  if (a_handle) {
    lt_dlclose(a_handle);

    for (std::vector<lt_dlhandle>::iterator I = OpenedHandles.begin(),
         E = OpenedHandles.end(); I != E; ++I) {
      if (*I == a_handle) {
        // Note: don't use the swap/pop_back trick here. Order is important.
        OpenedHandles.erase(I);
      }
    }
  }
}

void DynamicLibrary::LoadLibraryPermanently(const char* filename) {
  check_ltdl_initialization();
  lt_dlhandle a_handle = lt_dlopen(filename);

  if (a_handle == 0)
    a_handle = lt_dlopenext(filename);

  if (a_handle == 0)
    throw std::string("Can't open :") + filename + ": " + lt_dlerror();

  lt_dlmakeresident(a_handle);

  OpenedHandles.push_back(a_handle);
}

void* DynamicLibrary::SearchForAddressOfSymbol(const char* symbolName) {
  check_ltdl_initialization();
  for (std::vector<lt_dlhandle>::iterator I = OpenedHandles.begin(),
       E = OpenedHandles.end(); I != E; ++I) {
    lt_ptr ptr = lt_dlsym(*I, symbolName);
    if (ptr)
      return ptr;
  }
  return 0;
}

void *DynamicLibrary::GetAddressOfSymbol(const char *symbolName) {
  assert(handle != 0 && "Invalid DynamicLibrary handle");
  return lt_dlsym((lt_dlhandle) handle, symbolName);
}

} // namespace llvm
