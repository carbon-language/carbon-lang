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

namespace llvm {

class TargetRegisterClass;

class SSARegMap {
  std::vector<const TargetRegisterClass*> RegClassMap;

  unsigned rescale(unsigned Reg) {
    return Reg - MRegisterInfo::FirstVirtualRegister;
  }

 public:
  const TargetRegisterClass* getRegClass(unsigned Reg) {
    unsigned actualReg = rescale(Reg);
    assert(actualReg < RegClassMap.size() && "Register out of bounds");
    return RegClassMap[actualReg];
  }

  /// createVirtualRegister - Create and return a new virtual register in the
  /// function with the specified register class.
  ///
  unsigned createVirtualRegister(const TargetRegisterClass *RegClass) {
    RegClassMap.push_back(RegClass);
    return RegClassMap.size()+MRegisterInfo::FirstVirtualRegister-1;
  }

  unsigned getNumVirtualRegs() const {
    return RegClassMap.size();
  }
};

} // End llvm namespace

#endif
