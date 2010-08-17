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

namespace {

struct ExplicitSymbolsDeleter {
  ~ExplicitSymbolsDeleter() {
    if (ExplicitSymbols)
      delete ExplicitSymbols;
  }
};

}

static ExplicitSymbolsDeleter Dummy;

void llvm::sys::DynamicLibrary::AddSymbol(const char* symbolName,
                                          void *symbolValue) {
  if (ExplicitSymbols == 0)
    ExplicitSymbols = new std::map<std::string, void*>();
  (*ExplicitSymbols)[symbolName] = symbolValue;
}

#ifdef LLVM_ON_WIN32

#include "Win32/DynamicLibrary.inc"

#else

#if HAVE_DLFCN_H
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
#ifdef __CYGWIN__
  // Cygwin searches symbols only in the main
  // with the handle of dlopen(NULL, RTLD_GLOBAL).
  if (Filename == NULL)
    H = RTLD_DEFAULT;
#endif
  if (OpenedHandles == 0)
    OpenedHandles = new std::vector<void *>();
  OpenedHandles->push_back(H);
  return false;
}
#else

using namespace llvm;
using namespace llvm::sys;

bool DynamicLibrary::LoadLibraryPermanently(const char *Filename,
                                            std::string *ErrMsg) {
  if (ErrMsg) *ErrMsg = "dlopen() not supported on this platform";
  return true;
}
#endif

namespace llvm {
void *SearchForAddressOfSpecialSymbol(const char* symbolName);
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

#if HAVE_DLFCN_H
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
#endif

  if (void *Result = llvm::SearchForAddressOfSpecialSymbol(symbolName))
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
