//===-- llvm/CodeGen/Sparc.h - Sparc Target Description ----------*- C++ -*--=//
//
// This file defines the Sparc processor targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPARC_H
#define LLVM_CODEGEN_SPARC_H

class TargetMachine;

// allocateSparcTargetMachine - Allocate and return a subclass of TargetMachine
// that implements the Sparc backend.
//
TargetMachine *allocateSparcTargetMachine();

#endif
