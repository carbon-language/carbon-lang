//===-- llvm/Target/TargetMachineImpls.h - Target Descriptions --*- C++ -*-===//
//
// This file defines the entry point to getting access to the various target
// machine implementations available to LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINEIMPLS_H
#define LLVM_TARGET_TARGETMACHINEIMPLS_H

class TargetMachine;

// allocateSparcTargetMachine - Allocate and return a subclass of TargetMachine
// that implements the Sparc backend.
//
TargetMachine *allocateSparcTargetMachine();

// allocateX86TargetMachine - Allocate and return a subclass of TargetMachine
// that implements the X86 backend.
//
TargetMachine *allocateX86TargetMachine();

#endif
