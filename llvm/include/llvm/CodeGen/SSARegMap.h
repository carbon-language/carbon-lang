//===-- llvm/CodeGen/SSARegMap.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Map register numbers to register classes that are correctly sized (typed) to
// hold the information. Assists register allocation. Contained by
// MachineFunction, should be deleted by register allocator when it is no
// longer needed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SSAREGMAP_H
#define LLVM_CODEGEN_SSAREGMAP_H

#include "llvm/Target/MRegisterInfo.h"
#include "Support/DenseMap.h"

namespace llvm {

class TargetRegisterClass;

class SSARegMap {
  DenseMap<const TargetRegisterClass*, VirtReg2IndexFunctor> RegClassMap;
  unsigned NextRegNum;

 public:
  SSARegMap() : NextRegNum(MRegisterInfo::FirstVirtualRegister) { }

  const TargetRegisterClass* getRegClass(unsigned Reg) {
    return RegClassMap[Reg];
  }

  /// createVirtualRegister - Create and return a new virtual register in the
  /// function with the specified register class.
  ///
  unsigned createVirtualRegister(const TargetRegisterClass *RegClass) {
    RegClassMap.grow(NextRegNum);
    RegClassMap[NextRegNum] = RegClass;
    return NextRegNum++;
  }

  unsigned getLastVirtReg() const {
    return NextRegNum - 1;
  }
};

} // End llvm namespace

#endif
