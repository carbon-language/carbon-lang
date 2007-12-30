//===-- llvm/CodeGen/SSARegMap.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSARegMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SSAREGMAP_H
#define LLVM_CODEGEN_SSAREGMAP_H

#include "llvm/Target/MRegisterInfo.h"
#include <vector>

namespace llvm {
  
/// SSARegMap - Keep track of information for each virtual register, including
/// its register class.
class SSARegMap {
  /// VRegInfo - Information we keep for each virtual register.  The entries in
  /// this vector are actually converted to vreg numbers by adding the 
  /// MRegisterInfo::FirstVirtualRegister delta to their index.
  std::vector<const TargetRegisterClass*> VRegInfo;
  
public:
  SSARegMap() {
    VRegInfo.reserve(256);
  }

  /// getRegClass - Return the register class of the specified virtual register.
  const TargetRegisterClass *getRegClass(unsigned Reg) {
    Reg -= MRegisterInfo::FirstVirtualRegister;
    assert(Reg < VRegInfo.size() && "Invalid vreg!");
    return VRegInfo[Reg];
  }

  /// createVirtualRegister - Create and return a new virtual register in the
  /// function with the specified register class.
  ///
  unsigned createVirtualRegister(const TargetRegisterClass *RegClass) {
    assert(RegClass && "Cannot create register without RegClass!");
    VRegInfo.push_back(RegClass);
    return getLastVirtReg();
  }

  /// getLastVirtReg - Return the highest currently assigned virtual register.
  ///
  unsigned getLastVirtReg() const {
    return VRegInfo.size()+MRegisterInfo::FirstVirtualRegister-1;
  }
};

} // End llvm namespace

#endif
