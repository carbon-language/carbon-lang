//===-- Intercept.cpp - System function interception routines -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// If a function call occurs to an external function, the JIT is designed to use
// the dynamic loader interface to find a function to call.  This is useful for
// calling system calls and library functions that are not available in LLVM.
// Some system calls, however, need to be handled specially.  For this reason,
// we intercept some of them here and use our own stubs to handle them.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/Config/config.h"
#include <iostream>
using namespace llvm;

// AtExitHandlers - List of functions to call when the program exits,
// registered with the atexit() library function.
static std::vector<void (*)()> AtExitHandlers;

/// runAtExitHandlers - Run any functions registered by the program's
/// calls to atexit(3), which we intercept and store in
/// AtExitHandlers.
///
static void runAtExitHandlers() {
  while (!AtExitHandlers.empty()) {
    void (*Fn)() = AtExitHandlers.back();
    AtExitHandlers.pop_back();
    Fn();
  }
}

//===----------------------------------------------------------------------===//
// Function stubs that are invoked instead of certain library calls
//===----------------------------------------------------------------------===//

// Force the following functions to be linked in to anything that uses the
// JIT. This is a hack designed to work around the all-too-clever Glibc
// strategy of making these functions work differently when inlined vs. when
// not inlined, and hiding their real definitions in a separate archive file
// that the dynamic linker can't see. For more info, search for
// 'libc_nonshared.a' on Google, or read http://llvm.org/PR274.
#if defined(__linux__)
#if defined(HAVE_SYS_STAT_H)
#include <sys/stat.h>
#endif
void *FunctionPointers[] = {
  (void *)(intptr_t) stat,
  (void *)(intptr_t) fstat,
  (void *)(intptr_t) lstat,
  (void *)(intptr_t) stat64,
  (void *)(intptr_t) fstat64,
  (void *)(intptr_t) lstat64,
  (void *)(intptr_t) atexit,
  (void *)(intptr_t) mknod
};
#endif // __linux__

// __mainFunc - If the program does not have a linked in __main function, allow
// it to run, but print a warning.
static void __mainFunc() {
  fprintf(stderr, "WARNING: Program called __main but was not linked to "
          "libcrtend.a.\nThis probably won't hurt anything unless the "
          "program is written in C++.\n");
}

// jit_exit - Used to intercept the "exit" library call.
static void jit_exit(int Status) {
  runAtExitHandlers();   // Run atexit handlers...
  exit(Status);
}

// jit_atexit - Used to intercept the "atexit" library call.
static int jit_atexit(void (*Fn)(void)) {
  AtExitHandlers.push_back(Fn);    // Take note of atexit handler...
  return 0;  // Always successful
}

//===----------------------------------------------------------------------===//
//
/// getPointerToNamedFunction - This method returns the address of the specified
/// function by using the dynamic loader interface.  As such it is only useful
/// for resolving library symbols, not code generated symbols.
///
void *JIT::getPointerToNamedFunction(const std::string &Name) {
  // Check to see if this is one of the functions we want to intercept.  Note,
  // we cast to intptr_t here to silence a -pedantic warning that complains
  // about casting a function pointer to a normal pointer.
  if (Name == "exit") return (void*)(intptr_t)&jit_exit;
  if (Name == "atexit") return (void*)(intptr_t)&jit_atexit;

  // If the program does not have a linked in __main function, allow it to run,
  // but print a warning.
  if (Name == "__main") return (void*)(intptr_t)&__mainFunc;

  const char *NameStr = Name.c_str();
  // If this is an asm specifier, skip the sentinal.
  if (NameStr[0] == 1) ++NameStr;
  
  // If it's an external function, look it up in the process image...
  void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr);
  if (Ptr) return Ptr;
  
  // If it wasn't found and if it starts with an underscore ('_') character, and
  // has an asm specifier, try again without the underscore.
  if (Name[0] == 1 && NameStr[0] == '_') {
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr+1);
    if (Ptr) return Ptr;
  }

  std::cerr << "ERROR: Program used external function '" << Name
            << "' which could not be resolved!\n";
  abort();
  return 0;
}
