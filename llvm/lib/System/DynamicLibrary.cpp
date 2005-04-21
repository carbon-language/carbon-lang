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
#include "llvm/Config/config.h"

// It is not possible to use ltdl.c on VC++ builds as the terms of its LGPL
// license and special exception would cause all of LLVM to be placed under
// the LGPL.  This is because the exception applies only when libtool is
// used, and obviously libtool is not used with Visual Studio.  An entirely
// separate implementation is provided in win32/DynamicLibrary.cpp.

#ifdef LLVM_ON_WIN32

#include "Win32/DynamicLibrary.inc"

#else

#include "ltdl.h"
#include <cassert>
using namespace llvm;
using namespace llvm::sys;

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

  // If this is darwin, it has some funky issues, try to solve them here.  Some
  // important symbols are marked 'private external' which doesn't allow
  // SearchForAddressOfSymbol to find them.  As such, we special case them here,
  // there is only a small handful of them.
#ifdef __APPLE__
  {
#define EXPLICIT_SYMBOL(SYM) \
   extern void *SYM; if (!strcmp(symbolName, #SYM)) return &SYM
    EXPLICIT_SYMBOL(__ashldi3);
    EXPLICIT_SYMBOL(__ashrdi3);
    EXPLICIT_SYMBOL(__cmpdi2);
    EXPLICIT_SYMBOL(__divdi3);
    EXPLICIT_SYMBOL(__eprintf);
    EXPLICIT_SYMBOL(__fixdfdi);
    EXPLICIT_SYMBOL(__fixsfdi);
    EXPLICIT_SYMBOL(__fixunsdfdi);
    EXPLICIT_SYMBOL(__fixunssfdi);
    EXPLICIT_SYMBOL(__floatdidf);
    EXPLICIT_SYMBOL(__floatdisf);
    EXPLICIT_SYMBOL(__lshrdi3);
    EXPLICIT_SYMBOL(__moddi3);
    EXPLICIT_SYMBOL(__udivdi3);
    EXPLICIT_SYMBOL(__umoddi3);
#undef EXPLICIT_SYMBOL
  }
#endif

  return 0;
}

void *DynamicLibrary::GetAddressOfSymbol(const char *symbolName) {
  assert(handle != 0 && "Invalid DynamicLibrary handle");
  return lt_dlsym((lt_dlhandle) handle, symbolName);
}

#endif // LLVM_ON_WIN32
