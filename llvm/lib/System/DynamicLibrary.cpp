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
    extern void *__ashldi3;    if (Name == "__ashldi3")    return &__ashldi3;
    extern void *__ashrdi3;    if (Name == "__ashrdi3")    return &__ashrdi3;
    extern void *__cmpdi2;     if (Name == "__cmpdi2")     return &__cmpdi2;
    extern void *__divdi3;     if (Name == "__divdi3")     return &__divdi3;
    extern void *__eprintf;    if (Name == "__eprintf")    return &__eprintf;
    extern void *__fixdfdi;    if (Name == "__fixdfdi")    return &__fixdfdi;
    extern void *__fixsfdi;    if (Name == "__fixsfdi")    return &__fixsfdi;
    extern void *__fixunsdfdi; if (Name == "__fixunsdfdi") return &__fixunsdfdi;
    extern void *__fixunssfdi; if (Name == "__fixunssfdi") return &__fixunssfdi;
    extern void *__floatdidf;  if (Name == "__floatdidf")  return &__floatdidf;
    extern void *__floatdisf;  if (Name == "__floatdisf")  return &__floatdisf;
    extern void *__lshrdi3;    if (Name == "__lshrdi3")    return &__lshrdi3;
    extern void *__moddi3;     if (Name == "__moddi3")     return &__moddi3;
    extern void *__udivdi3;    if (Name == "__udivdi3")    return &__udivdi3;
    extern void *__umoddi3;    if (Name == "__umoddi3")    return &__umoddi3;
  }
#endif

  return 0;
}

void *DynamicLibrary::GetAddressOfSymbol(const char *symbolName) {
  assert(handle != 0 && "Invalid DynamicLibrary handle");
  return lt_dlsym((lt_dlhandle) handle, symbolName);
}

