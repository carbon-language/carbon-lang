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
#include <map>

namespace llvm {

class AMDGPUMachineFunction : public MachineFunctionInfo {
  virtual void anchor();
public:
  AMDGPUMachineFunction(const MachineFunction &MF);
  unsigned ShaderType;
  /// A map to keep track of local memory objects and their offsets within
  /// the local memory space.
  std::map<const GlobalValue *, unsigned> LocalMemoryObjects;
  /// Number of bytes in the LDS that are being used.
  unsigned LDSSize;
};

}
#endif // AMDGPUMACHINEFUNCTION_H
