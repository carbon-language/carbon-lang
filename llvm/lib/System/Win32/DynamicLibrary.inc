//===- Win32/DynamicLibrary.cpp - Win32 DL Implementation -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of DynamicLibrary.
//
//===----------------------------------------------------------------------===//

#include "Win32.h"

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
#endif

#ifdef __MINGW32__
 #if (HAVE_LIBIMAGEHLP != 1)
  #error "libimagehlp.a should be present"
 #endif
#else
 #pragma comment(lib, "dbghelp.lib")
#endif

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be UNIX code.
//===----------------------------------------------------------------------===//

static std::vector<HMODULE> OpenedHandles;

extern "C" {
  static BOOL CALLBACK ELM_Callback(PSTR  ModuleName,
                                    ULONG ModuleBase,
                                    ULONG ModuleSize,
                                    PVOID UserContext)
  {
    // Ignore VC++ runtimes prior to 7.1.  Somehow some of them get loaded
    // into the process.
    if (stricmp(ModuleName, "msvci70") != 0 &&
        stricmp(ModuleName, "msvcirt") != 0 &&
        stricmp(ModuleName, "msvcp50") != 0 &&
        stricmp(ModuleName, "msvcp60") != 0 &&
        stricmp(ModuleName, "msvcp70") != 0 &&
        stricmp(ModuleName, "msvcr70") != 0 &&
        stricmp(ModuleName, "msvcrt") != 0 &&
        stricmp(ModuleName, "msvcrt20") != 0 &&
        stricmp(ModuleName, "msvcrt40") != 0) {
      OpenedHandles.push_back((HMODULE)ModuleBase);
    }
    return TRUE;
  }
}

DynamicLibrary::DynamicLibrary() : handle(0) {
  handle = GetModuleHandle(NULL);
  OpenedHandles.push_back((HMODULE)handle);
}

DynamicLibrary::DynamicLibrary(const char*filename) : handle(0) {
  HMODULE a_handle = LoadLibrary(filename);

  if (a_handle == 0)
    ThrowError(std::string(filename) + ": Can't open : ");

  handle = a_handle;
  OpenedHandles.push_back(a_handle);
}

DynamicLibrary::~DynamicLibrary() {
  if (handle == 0)
    return;

  // GetModuleHandle() does not increment the ref count, so we must not free
  // the handle to the executable.
  if (handle != GetModuleHandle(NULL))
    FreeLibrary((HMODULE)handle);
  handle = 0;

  for (std::vector<HMODULE>::iterator I = OpenedHandles.begin(),
       E = OpenedHandles.end(); I != E; ++I) {
    if (*I == handle) {
      // Note: don't use the swap/pop_back trick here. Order is important.
      OpenedHandles.erase(I);
    }
  }
}

bool DynamicLibrary::LoadLibraryPermanently(const char *filename,
                                            std::string *ErrMsg) {
  if (filename) {
    HMODULE a_handle = LoadLibrary(filename);

    if (a_handle == 0)
      return GetError(std::string(filename) + ": Can't open : ", ErrMsg);

    OpenedHandles.push_back(a_handle);
  } else {
    // When no file is specified, enumerate all DLLs and EXEs in the
    // process.
    EnumerateLoadedModules(GetCurrentProcess(), ELM_Callback, 0);
  }

  // Because we don't remember the handle, we will never free it; hence,
  // it is loaded permanently.
  return false;
}

void* DynamicLibrary::SearchForAddressOfSymbol(const char* symbolName) {
  // First check symbols added via AddSymbol().
  std::map<std::string, void *>::iterator I = g_symbols.find(symbolName);
  if (I != g_symbols.end())
    return I->second;

  // Now search the libraries.
  for (std::vector<HMODULE>::iterator I = OpenedHandles.begin(),
       E = OpenedHandles.end(); I != E; ++I) {
    FARPROC ptr = GetProcAddress((HMODULE)*I, symbolName);
    if (ptr)
      return (void *) ptr;
  }

  return 0;
}

void *DynamicLibrary::GetAddressOfSymbol(const char *symbolName) {
  assert(handle != 0 && "Invalid DynamicLibrary handle");
  return (void *) GetProcAddress((HMODULE)handle, symbolName);
}

}

