//===-- DynamicLinker.cpp - Implement DynamicLinker interface -------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Lightweight interface to dynamic library linking and loading, and dynamic
// symbol lookup functionality, in whatever form the operating system
// provides it.
//
// Possible future extensions include support for the HPUX shl_load()
// interface, the Mac OS X NSLinkModule() interface, and the Windows
// LoadLibrary() interface.
//
// Note that we assume that if dlopen() is available, then dlsym() is too.
//
//===----------------------------------------------------------------------===//

#include "Support/DynamicLinker.h"
#include "Config/dlfcn.h"
#include "Config/windows.h"
#include <cassert>
using namespace llvm;

bool llvm::LinkDynamicObject (const char *filename, std::string *ErrorMessage) {
#if defined (HAVE_DLOPEN)
  if (dlopen (filename, RTLD_NOW | RTLD_GLOBAL) == 0) {
    if (ErrorMessage) *ErrorMessage = dlerror ();
    return true;
  }
  return false;
#elif defined(HAVE_WINDOWS_H)
  if (LoadLibrary(filename))
    return false;
  if (ErrorMessage) {
    char Buffer[100];
    // FIXME: This should use FormatMessage
    sprintf(Buffer, "Windows error code %d\n", GetLastError());
    *ErrorMessage = Buffer;
  }
  return true;
#else
  assert (0 && "Dynamic object linking not implemented for this platform");
#endif
}

void *llvm::GetAddressOfSymbol (const char *symbolName) {
#if defined (HAVE_DLOPEN)
#  ifdef RTLD_DEFAULT
    return dlsym (RTLD_DEFAULT, symbolName);
#  else
    static void* CurHandle = dlopen(0, RTLD_LAZY);
    return dlsym(CurHandle, symbolName);
#  endif
#elif defined(HAVE_WINDOWS_H)
  static HMODULE ModHandle = NULL;
  if (ModHandle == 0) ModHandle = GetModuleHandle(NULL);
  return (void*)GetProcAddress(ModHandle, symbolName);
#else
  assert (0 && "Dynamic symbol lookup not implemented for this platform");
#endif
}

// soft, cushiony C++ interface.
void *llvm::GetAddressOfSymbol(const std::string &symbolName) {
  return GetAddressOfSymbol(symbolName.c_str());
}
