//===- TargetSelect.h - Target Selection & Registration ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities to make sure that certain classes of targets are
// linked into the main application executable, and initialize them as
// appropriate.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSELECT_H
#define LLVM_TARGET_TARGETSELECT_H

#include "llvm/Config/config.h"

extern "C" {
  // Declare all of the target-initialization functions that are available.
#define LLVM_TARGET(TargetName) void LLVMInitialize##TargetName##TargetInfo();
#include "llvm/Config/Targets.def"

#define LLVM_TARGET(TargetName) void LLVMInitialize##TargetName##Target();
#include "llvm/Config/Targets.def"
  
  // Declare all of the available asm-printer initialization functions.
#define LLVM_ASM_PRINTER(TargetName) void LLVMInitialize##TargetName##AsmPrinter();
#include "llvm/Config/AsmPrinters.def"
}

namespace llvm {
  /// InitializeAllTargets - The main program should call this function if it
  /// wants access to all available targets that LLVM is configured to
  /// support. This allows the client to query the available targets using the
  /// target registration mechanisms.
  inline void InitializeAllTargets() {
#define LLVM_TARGET(TargetName) LLVMInitialize##TargetName##TargetInfo();
#include "llvm/Config/Targets.def"

#define LLVM_TARGET(TargetName) LLVMInitialize##TargetName##Target();
#include "llvm/Config/Targets.def"
  }
  
  /// InitializeAllAsmPrinters - The main program should call this function if
  /// it wants all asm printers that LLVM is configured to support.  This will
  /// cause them to be linked into its executable.
  inline void InitializeAllAsmPrinters() {
#define LLVM_ASM_PRINTER(TargetName) LLVMInitialize##TargetName##AsmPrinter();
#include "llvm/Config/AsmPrinters.def"
  }
  
  /// InitializeNativeTarget - The main program should call this function to
  /// initialize the native target corresponding to the host.  This is useful 
  /// for JIT applications to ensure that the target gets linked in correctly.
  inline bool InitializeNativeTarget() {
  // If we have a native target, initialize it to ensure it is linked in.
#ifdef LLVM_NATIVE_ARCH
#define DoInit2(TARG) \
    LLVMInitialize ## TARG ## Info ();          \
    LLVMInitialize ## TARG ()
#define DoInit(T) DoInit2(T)
    DoInit(LLVM_NATIVE_ARCH);
    return false;
#undef DoInit
#undef DoInit2
#else
    return true;
#endif
  }  
}

#endif
