//===-- llvm/Target/TargetMachineImpls.h - Target Descriptions --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the entry point to getting access to the various target
// machine implementations available to LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINEIMPLS_H
#define LLVM_TARGET_TARGETMACHINEIMPLS_H

class TargetMachine;
class Module;

// allocateSparcTargetMachine - Allocate and return a subclass of TargetMachine
// that implements the Sparc backend.
//
TargetMachine *allocateSparcTargetMachine(const Module &M);

// allocateX86TargetMachine - Allocate and return a subclass of TargetMachine
// that implements the X86 backend.  The X86 target machine can run in
// "emulation" mode, where it is capable of emulating machines of larger pointer
// size and different endianness if desired.
//
TargetMachine *allocateX86TargetMachine(const Module &M);

#endif
