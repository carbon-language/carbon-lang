//===-- DynamicLibrary.cpp - Runtime link/load libraries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system DynamicLibrary concept.
//
// FIXME: This file leaks the ExplicitSymbols and OpenedHandles vector, and is
// not thread safe!
//
//===----------------------------------------------------------------------===//

#include "llvm/System/DynamicLibrary.h"
#include "llvm/Config/config.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

// Collection of symbol name/value pairs to be searched prior to any libraries.
static std::map<std::string, void*> *ExplicitSymbols = 0;

static struct ExplicitSymbolsDeleter {
  ~ExplicitSymbolsDeleter() {
    if (ExplicitSymbols)
      delete ExplicitSymbols;
  }
} Dummy;

void llvm::sys::DynamicLibrary::AddSymbol(const char* symbolName,
                                          void *symbolValue) {
  if (ExplicitSymbols == 0)
    ExplicitSymbols = new std::map<std::string, void*>();
  (*ExplicitSymbols)[symbolName] = symbolValue;
}

#ifdef LLVM_ON_WIN32

#include "Win32/DynamicLibrary.inc"

#else

#include <dlfcn.h>
using namespace llvm;
using namespace llvm::sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

static std::vector<void *> *OpenedHandles = 0;


bool DynamicLibrary::LoadLibraryPermanently(const char *Filename,
                                            std::string *ErrMsg) {
  void *H = dlopen(Filename, RTLD_LAZY|RTLD_GLOBAL);
  if (H == 0) {
    if (ErrMsg) *ErrMsg = dlerror();
    return true;
  }
  if (OpenedHandles == 0)
    OpenedHandles = new std::vector<void *>();
  OpenedHandles->push_back(H);
  return false;
}

static void *SearchForAddressOfSpecialSymbol(const char* symbolName) {
#define EXPLICIT_SYMBOL(SYM) \
   extern void *SYM; if (!strcmp(symbolName, #SYM)) return &SYM

  // If this is darwin, it has some funky issues, try to solve them here.  Some
  // important symbols are marked 'private external' which doesn't allow
  // SearchForAddressOfSymbol to find them.  As such, we special case them here,
  // there is only a small handful of them.

#ifdef __APPLE__
  {
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
  }
#endif

#ifdef __CYGWIN__
  {
    EXPLICIT_SYMBOL(_alloca);
    EXPLICIT_SYMBOL(__main);
  }
#endif

#undef EXPLICIT_SYMBOL
  return 0;
}

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
  if (OpenedHandles) {
    for (std::vector<void *>::iterator I = OpenedHandles->begin(),
         E = OpenedHandles->end(); I != E; ++I) {
      //lt_ptr ptr = lt_dlsym(*I, symbolName);
      void *ptr = dlsym(*I, symbolName);
      if (ptr) {
        return ptr;
      }
    }
  }

  if (void *Result = SearchForAddressOfSpecialSymbol(symbolName))
    return Result;

// This macro returns the address of a well-known, explicit symbol
#define EXPLICIT_SYMBOL(SYM) \
   if (!strcmp(symbolName, #SYM)) return &SYM

// On linux we have a weird situation. The stderr/out/in symbols are both
// macros and global variables because of standards requirements. So, we 
// boldly use the EXPLICIT_SYMBOL macro without checking for a #define first.
#if defined(__linux__)
  {
    EXPLICIT_SYMBOL(stderr);
    EXPLICIT_SYMBOL(stdout);
    EXPLICIT_SYMBOL(stdin);
  }
#else
  // For everything else, we want to check to make sure the symbol isn't defined
  // as a macro before using EXPLICIT_SYMBOL.
  {
#ifndef stdin
    EXPLICIT_SYMBOL(stdin);
#endif
#ifndef stdout
    EXPLICIT_SYMBOL(stdout);
#endif
#ifndef stderr
    EXPLICIT_SYMBOL(stderr);
#endif
  }
#endif
#undef EXPLICIT_SYMBOL

  return 0;
}

#endif // LLVM_ON_WIN32
