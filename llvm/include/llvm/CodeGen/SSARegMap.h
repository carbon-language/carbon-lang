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
#include "llvm/ADT/IndexedMap.h"

namespace llvm {

class TargetRegisterClass;

class SSARegMap {
  IndexedMap<const TargetRegisterClass*, VirtReg2IndexFunctor> RegClassMap;
  IndexedMap<std::pair<unsigned, unsigned>, VirtReg2IndexFunctor> RegSubIdxMap;
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
    assert(RegClass && "Cannot create register without RegClass!");
    RegClassMap.grow(NextRegNum);
    RegClassMap[NextRegNum] = RegClass;
    RegSubIdxMap.grow(NextRegNum);
    RegSubIdxMap[NextRegNum] = std::make_pair(0,0);
    return NextRegNum++;
  }

  unsigned getLastVirtReg() const {
    return NextRegNum - 1;
  }

  void setIsSubRegister(unsigned Reg, unsigned SuperReg, unsigned SubIdx) {
    RegSubIdxMap[Reg] = std::make_pair(SuperReg, SubIdx);
  }

  bool isSubRegister(unsigned Reg) const {
    return RegSubIdxMap[Reg].first != 0;
  }

  unsigned getSuperRegister(unsigned Reg) const {
    return RegSubIdxMap[Reg].first;
  }

  unsigned getSubRegisterIndex(unsigned Reg) const {
    return RegSubIdxMap[Reg].second;
  }
};

} // End llvm namespace

#endif
