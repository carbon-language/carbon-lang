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
// interface, and the Mac OS X NSLinkModule() interface.
//
// Note that we assume that if dlopen() is available, then dlsym() is too.
//
//===----------------------------------------------------------------------===//

#include "Support/DynamicLinker.h"
#include "Config/dlfcn.h"
#include "Config/windows.h"
#include <cassert>
#include <vector>
using namespace llvm;

#if defined(HAVE_WINDOWS_H)
// getLoadedLibs - Keep track of the shared objects that are loaded into the
// process address space, as the windows GetProcAddress function does not
// automatically search an entire address space, it only searches a specific
// object.
static std::vector<HMODULE> &getLoadedLibHandles() {
  static std::vector<HMODULE> *LoadedLibHandles = 0;
  if (LoadedLibHandles == 0) {
    LoadedLibHandles = new std::vector<HMODULE>();
    if (HMODULE H = GetModuleHandle(NULL))                // JIT symbols
      LoadedLibHandles->push_back(H);
    if (HMODULE MH = GetModuleHandle("cygwin1.dll"))      // Cygwin symbols OR
      LoadedLibHandles->push_back(MH);
    else if (HMODULE MH = GetModuleHandle("msvcr80.dll")) // VC++ symbols
      LoadedLibHandles->push_back(MH);
  }
  return *LoadedLibHandles;
}
#endif

bool llvm::LinkDynamicObject(const char *filename, std::string *ErrorMessage) {
#if defined(HAVE_WINDOWS_H)
  if (HMODULE Handle = LoadLibrary(filename)) {
    // Allow GetProcAddress in this module
    getLoadedLibHandles().push_back(Handle);
    return false;
  }
  if (ErrorMessage) {
    char Buffer[100];
    // FIXME: This should use FormatMessage
    sprintf(Buffer, "Windows error code %d\n", GetLastError());
    *ErrorMessage = Buffer;
  }
  return true;
#elif defined (HAVE_DLOPEN)
  if (dlopen (filename, RTLD_NOW | RTLD_GLOBAL) == 0) {
    if (ErrorMessage) *ErrorMessage = dlerror ();
    return true;
  }
  return false;
#else
  assert (0 && "Dynamic object linking not implemented for this platform");
#endif
}

void *llvm::GetAddressOfSymbol(const char *symbolName) {
#if defined(HAVE_WINDOWS_H)
  std::vector<HMODULE> &LH = getLoadedLibHandles();
  for (unsigned i = 0, e = LH.size(); i != e; ++i)
    if (void *Val = (void*)GetProcAddress(LH[i], symbolName))
      return Val;
  return 0;
#elif defined(HAVE_DLOPEN)
#  ifdef RTLD_DEFAULT
    return dlsym (RTLD_DEFAULT, symbolName);
#  else
    static void* CurHandle = dlopen(0, RTLD_LAZY);
    return dlsym(CurHandle, symbolName);
#  endif
#else
  assert (0 && "Dynamic symbol lookup not implemented for this platform");
#endif
}

// soft, cushiony C++ interface.
void *llvm::GetAddressOfSymbol(const std::string &symbolName) {
  return GetAddressOfSymbol(symbolName.c_str());
}
