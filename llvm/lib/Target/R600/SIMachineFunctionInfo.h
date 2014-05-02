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

  struct RegSpillTracker {
  private:
    unsigned CurrentLane;
    std::map<unsigned, SpilledReg> SpilledRegisters;
  public:
    unsigned LaneVGPR;
    RegSpillTracker() : CurrentLane(0), SpilledRegisters(), LaneVGPR(0) { }
    /// \p NumRegs The number of consecutive registers what need to be spilled.
    ///            This function will ensure that all registers are stored in
    ///            the same VGPR.
    /// \returns The lane to be used for storing the first register.
    unsigned reserveLanes(MachineRegisterInfo &MRI, MachineFunction *MF,
                          unsigned NumRegs = 1);
    void addSpilledReg(unsigned FrameIndex, unsigned Reg, int Lane = -1);
    const SpilledReg& getSpilledReg(unsigned FrameIndex);
    bool programSpillsRegisters() { return !SpilledRegisters.empty(); }
  };

  // SIMachineFunctionInfo definition

  SIMachineFunctionInfo(const MachineFunction &MF);
  unsigned PSInputAddr;
  struct RegSpillTracker SpillTracker;
};

} // End namespace llvm


#endif //_SIMACHINEFUNCTIONINFO_H_
