//===--- HexagonBitTracker.h ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBITTRACKER_H
#define HEXAGONBITTRACKER_H

#include "BitTracker.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class HexagonInstrInfo;
  class HexagonRegisterInfo;
}

struct HexagonEvaluator : public BitTracker::MachineEvaluator {
  typedef BitTracker::CellMapType CellMapType;
  typedef BitTracker::RegisterRef RegisterRef;
  typedef BitTracker::RegisterCell RegisterCell;
  typedef BitTracker::BranchTargetList BranchTargetList;

  HexagonEvaluator(const llvm::HexagonRegisterInfo &tri,
        llvm::MachineRegisterInfo &mri, const llvm::HexagonInstrInfo &tii,
        llvm::MachineFunction &mf);

  virtual bool evaluate(const llvm::MachineInstr *MI,
        const CellMapType &Inputs, CellMapType &Outputs) const;
  virtual bool evaluate(const llvm::MachineInstr *BI,
        const CellMapType &Inputs, BranchTargetList &Targets,
        bool &FallsThru) const;

  virtual BitTracker::BitMask mask(unsigned Reg, unsigned Sub) const;

  llvm::MachineFunction &MF;
  llvm::MachineFrameInfo &MFI;
  const llvm::HexagonInstrInfo &TII;

private:
  bool evaluateLoad(const llvm::MachineInstr *MI, const CellMapType &Inputs,
        CellMapType &Outputs) const;
  bool evaluateFormalCopy(const llvm::MachineInstr *MI,
        const CellMapType &Inputs, CellMapType &Outputs) const;

  unsigned getNextPhysReg(unsigned PReg, unsigned Width) const;
  unsigned getVirtRegFor(unsigned PReg) const;

  // Type of formal parameter extension.
  struct ExtType {
    enum { SExt, ZExt };
    char Type;
    uint16_t Width;
    ExtType() : Type(0), Width(0) {}
    ExtType(char t, uint16_t w) : Type(t), Width(w) {}
  };
  // Map VR -> extension type.
  typedef llvm::DenseMap<unsigned,ExtType> RegExtMap;
  RegExtMap VRX;
};

#endif

