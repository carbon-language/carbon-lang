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


#ifndef LLVM_LIB_TARGET_R600_SIMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_R600_SIMACHINEFUNCTIONINFO_H

#include "AMDGPUMachineFunction.h"
#include <map>

namespace llvm {

class MachineRegisterInfo;

/// This class keeps track of the SPI_SP_INPUT_ADDR config register, which
/// tells the hardware which interpolation parameters to load.
class SIMachineFunctionInfo : public AMDGPUMachineFunction {
  void anchor() override;
public:

  struct SpilledReg {
    unsigned VGPR;
    int Lane;
    SpilledReg(unsigned R, int L) : VGPR (R), Lane (L) { }
    SpilledReg() : VGPR(0), Lane(-1) { }
    bool hasLane() { return Lane != -1;}
  };

  // SIMachineFunctionInfo definition

  SIMachineFunctionInfo(const MachineFunction &MF);
  SpilledReg getSpilledReg(MachineFunction *MF, unsigned FrameIndex,
                           unsigned SubIdx);
  unsigned PSInputAddr;
  unsigned NumUserSGPRs;
  std::map<unsigned, unsigned> LaneVGPRs;
};

} // End namespace llvm


#endif
