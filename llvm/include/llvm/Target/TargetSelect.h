//===- TargetSelect.h - Target Selection & Registration -------------------===//
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

namespace llvm {
  // Declare all of the target-initialization functions that are available.
#define LLVM_TARGET(TargetName) void Initialize##TargetName##Target();
#include "llvm/Config/Targets.def"
  
  // Declare all of the available asm-printer initialization functions.
  // Declare all of the target-initialization functions.
#define LLVM_ASM_PRINTER(TargetName) void Initialize##TargetName##AsmPrinter();
#include "llvm/Config/AsmPrinters.def"
  
  /// InitializeAllTargets - The main program should call this function if it
  /// wants to link in all available targets that LLVM is configured to support.
  inline void InitializeAllTargets() {
#define LLVM_TARGET(TargetName) llvm::Initialize##TargetName##Target();
#include "llvm/Config/Targets.def"
  }
  
  /// InitializeAllAsmPrinters - The main program should call this function if
  /// it wants all asm printers that LLVM is configured to support.  This will
  /// cause them to be linked into its executable.
  inline void InitializeAllAsmPrinters() {
#define LLVM_ASM_PRINTER(TargetName) Initialize##TargetName##AsmPrinter();
#include "llvm/Config/AsmPrinters.def"
  }
  
  
  /// InitializeNativeTarget - The main program should call this function to
  /// initialize the native target corresponding to the host.  This is useful 
  /// for JIT applications to ensure that the target gets linked in correctly.
  inline bool InitializeNativeTarget() {
  // If we have a native target, initialize it to ensure it is linked in.
#ifdef LLVM_NATIVE_ARCH
#define DoInit2(TARG, MOD)   llvm::Initialize ## TARG ## MOD()
#define DoInit(T, M) DoInit2(T, M)
    DoInit(LLVM_NATIVE_ARCH, Target);
    return false;
#undef DoInit
#undef DoInit2
#else
    return true;
#endif
  }  
}

#endif
