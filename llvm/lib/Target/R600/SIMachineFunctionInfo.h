//===- SIMachineFunctionInfo.h - SIMachineFunctionInfo interface -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//


#ifndef SIMACHINEFUNCTIONINFO_H_
#define SIMACHINEFUNCTIONINFO_H_

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// This class keeps track of the SPI_SP_INPUT_ADDR config register, which
/// tells the hardware which interpolation parameters to load.
class SIMachineFunctionInfo : public MachineFunctionInfo {
public:
  static const char *ShaderTypeAttribute;

  SIMachineFunctionInfo(const MachineFunction &MF);
  unsigned SPIPSInputAddr;
  unsigned ShaderType;
};

} // End namespace llvm


#endif //_SIMACHINEFUNCTIONINFO_H_
