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
#include <iostream>
#include <sys/stat.h>
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
// 'libc_nonshared.a' on Google, or read http://llvm.cs.uiuc.edu/PR274.
#if defined(__linux__)
void *FunctionPointers[] = {
  (void *) stat,
  (void *) fstat,
  (void *) lstat,
  (void *) stat64,
  (void *) fstat64,
  (void *) lstat64,
  (void *) atexit,
  (void *) mknod
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
  // Check to see if this is one of the functions we want to intercept...
  if (Name == "exit") return (void*)&jit_exit;
  if (Name == "atexit") return (void*)&jit_atexit;

  // If the program does not have a linked in __main function, allow it to run,
  // but print a warning.
  if (Name == "__main") return (void*)&__mainFunc;

  // If it's an external function, look it up in the process image...
  void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(Name);
  if (Ptr) return Ptr;

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

  std::cerr << "ERROR: Program used external function '" << Name
            << "' which could not be resolved!\n";
  abort();
  return 0;
}
