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
#include "Support/DynamicLinker.h"
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
void *FunctionPointers[] = {
  (void *) stat,
  (void *) stat64,
  (void *) fstat,
  (void *) fstat64,
  (void *) lstat,
  (void *) lstat64,
  (void *) atexit,
  (void *) mknod
};

// NoopFn - Used if we have nothing else to call...
static void NoopFn() {}

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

  // If it's an external function, look it up in the process image...
  void *Ptr = GetAddressOfSymbol(Name);
  if (Ptr == 0) {
    std::cerr << "WARNING: Cannot resolve fn '" << Name
	      << "' using a dummy noop function instead!\n";
    Ptr = (void*)NoopFn;
  }
  
  return Ptr;
}
