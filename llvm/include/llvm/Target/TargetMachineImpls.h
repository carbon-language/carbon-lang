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

namespace llvm {

  class TargetMachine;
  class Module;
  class IntrinsicLowering;
  
  // allocateCTargetMachine - Allocate and return a subclass of TargetMachine
  // that implements emits C code.  This takes ownership of the
  // IntrinsicLowering pointer, deleting it when the target machine is
  // destroyed.
  //
  TargetMachine *allocateCTargetMachine(const Module &M,
                                        IntrinsicLowering *IL = 0);

  // allocateSparcTargetMachine - Allocate and return a subclass of
  // TargetMachine that implements the Sparc backend.  This takes ownership of
  // the IntrinsicLowering pointer, deleting it when the target machine is
  // destroyed.
  //
  TargetMachine *allocateSparcTargetMachine(const Module &M,
                                            IntrinsicLowering *IL = 0);
  
  // allocateX86TargetMachine - Allocate and return a subclass of TargetMachine
  // that implements the X86 backend.  This takes ownership of the
  // IntrinsicLowering pointer, deleting it when the target machine is
  // destroyed.
  //
  TargetMachine *allocateX86TargetMachine(const Module &M,
                                          IntrinsicLowering *IL = 0);

  // allocatePowerPCTargetMachine - Allocate and return a subclass
  // of TargetMachine that implements the PowerPC backend.  This takes
  // ownership of the IntrinsicLowering pointer, deleting it when
  // the target machine is destroyed.
  //
  TargetMachine *allocatePowerPCTargetMachine(const Module &M,
                                              IntrinsicLowering *IL = 0);
} // End llvm namespace

#endif
