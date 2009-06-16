//===- llvm/InitializeAllAsmPrinters.h - Init Asm Printers ------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header initializes all assembler printers for all configured
// LLVM targets, ensuring that they are registered.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_INITIALIZE_ALL_ASM_PRINTERS_H
#define LLVM_INITIALIZE_ALL_ASM_PRINTERS_H

namespace llvm {

  // Declare all of the target-initialization functions.
#define LLVM_ASM_PRINTER(TargetName) void Initialize##TargetName##AsmPrinter();
#include "llvm/Config/AsmPrinters.def"

  namespace {
    struct InitializeAllAsmPrinters {
      InitializeAllAsmPrinters() {
        // Call all of the target-initialization functions.
#define LLVM_ASM_PRINTER(TargetName) llvm::Initialize##TargetName##AsmPrinter();
#include "llvm/Config/AsmPrinters.def"
      }
    } DoInitializeAllAsmPrinters;
  }
} // end namespace llvm

#endif // LLVM_INITIALIZE_ALL_ASM_PRINTERS_H
