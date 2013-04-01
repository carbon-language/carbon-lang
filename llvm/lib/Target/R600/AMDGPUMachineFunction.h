//===-- R600MachineFunctionInfo.h - R600 Machine Function Info ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef AMDGPUMACHINEFUNCTION_H
#define AMDGPUMACHINEFUNCTION_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class AMDGPUMachineFunction : public MachineFunctionInfo {
private:
  static const char *ShaderTypeAttribute;
public:
  AMDGPUMachineFunction(const MachineFunction &MF);
  unsigned ShaderType;
};

}
#endif // AMDGPUMACHINEFUNCTION_H
