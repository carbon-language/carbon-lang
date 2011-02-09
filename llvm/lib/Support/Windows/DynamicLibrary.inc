//===- Win32/DynamicLibrary.cpp - Win32 DL Implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of DynamicLibrary.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
#endif

#ifdef _MSC_VER
 #include <ntverp.h>
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

#ifdef _WIN64
  typedef DWORD64 ModuleBaseType;
#else
  typedef ULONG ModuleBaseType;
#endif

extern "C" {
// Use old callback if:
//  - Not using Visual Studio
//  - Visual Studio 2005 or earlier but only if we are not using the Windows SDK
//    or Windows SDK version is older than 6.0
// Use new callback if:
//  - Newer Visual Studio (comes with newer SDK).
//  - Visual Studio 2005 with Windows SDK 6.0+
#if defined(_MSC_VER)
  #if _MSC_VER < 1500 && (!defined(VER_PRODUCTBUILD) || VER_PRODUCTBUILD < 6000)
    #define OLD_ELM_CALLBACK_DECL 1
  #endif
#elif defined(__MINGW64__)
  // Use new callback.
#elif defined(__MINGW32__)
  #define OLD_ELM_CALLBACK_DECL 1
#endif

#ifdef OLD_ELM_CALLBACK_DECL
  static BOOL CALLBACK ELM_Callback(PSTR  ModuleName,
                                    ModuleBaseType ModuleBase,
                                    ULONG ModuleSize,
                                    PVOID UserContext)
#else
  static BOOL CALLBACK ELM_Callback(PCSTR  ModuleName,
                                    ModuleBaseType ModuleBase,
                                    ULONG ModuleSize,
                                    PVOID UserContext)
#endif
  {
    // Ignore VC++ runtimes prior to 7.1.  Somehow some of them get loaded
    // into the process.
    if (stricmp(ModuleName, "msvci70") != 0 &&
        stricmp(ModuleName, "msvcirt") != 0 &&
        stricmp(ModuleName, "msvcp50") != 0 &&
        stricmp(ModuleName, "msvcp60") != 0 &&
        stricmp(ModuleName, "msvcp70") != 0 &&
        stricmp(ModuleName, "msvcr70") != 0 &&
#ifndef __MINGW32__
        // Mingw32 uses msvcrt.dll by default. Don't ignore it.
        // Otherwise, user should be aware, what he's doing :)
        stricmp(ModuleName, "msvcrt") != 0 &&
#endif
        stricmp(ModuleName, "msvcrt20") != 0 &&
        stricmp(ModuleName, "msvcrt40") != 0) {
      OpenedHandles.push_back((HMODULE)ModuleBase);
    }
    return TRUE;
  }
}

bool DynamicLibrary::LoadLibraryPermanently(const char *filename,
                                            std::string *ErrMsg) {
  if (filename) {
    HMODULE a_handle = LoadLibrary(filename);

    if (a_handle == 0)
      return MakeErrMsg(ErrMsg, std::string(filename) + ": Can't open : ");

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

// Stack probing routines are in the support library (e.g. libgcc), but we don't
// have dynamic linking on windows. Provide a hook.
#define EXPLICIT_SYMBOL(SYM)                    \
  extern "C" { extern void *SYM; }
#define EXPLICIT_SYMBOL2(SYMFROM, SYMTO) EXPLICIT_SYMBOL(SYMTO)

#include "explicit_symbols.inc"

#undef EXPLICIT_SYMBOL
#undef EXPLICIT_SYMBOL2

void* DynamicLibrary::SearchForAddressOfSymbol(const char* symbolName) {
  // First check symbols added via AddSymbol().
  if (ExplicitSymbols) {
    std::map<std::string, void *>::iterator I =
      ExplicitSymbols->find(symbolName);
    std::map<std::string, void *>::iterator E = ExplicitSymbols->end();
    if (I != E)
      return I->second;
  }

  // Now search the libraries.
  for (std::vector<HMODULE>::iterator I = OpenedHandles.begin(),
       E = OpenedHandles.end(); I != E; ++I) {
    FARPROC ptr = GetProcAddress((HMODULE)*I, symbolName);
    if (ptr) {
      return (void *) ptr;
    }
  }

  #define EXPLICIT_SYMBOL(SYM)                    \
    if (!strcmp(symbolName, #SYM)) return (void*)&SYM;
  #define EXPLICIT_SYMBOL2(SYMFROM, SYMTO)        \
    if (!strcmp(symbolName, #SYMFROM)) return (void*)&SYMTO;

  {
    #include "explicit_symbols.inc"
  }

  #undef EXPLICIT_SYMBOL
  #undef EXPLICIT_SYMBOL2

  return 0;
}

}
