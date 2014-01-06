//===-- XCoreMachineFuctionInfo.h - XCore machine function info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares XCore-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef XCOREMACHINEFUNCTIONINFO_H
#define XCOREMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include <vector>

namespace llvm {

// Forward declarations
class Function;

/// XCoreFunctionInfo - This class is derived from MachineFunction private
/// XCore target-specific information for each MachineFunction.
class XCoreFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();
  bool LRSpillSlotSet;
  int LRSpillSlot;
  bool FPSpillSlotSet;
  int FPSpillSlot;
  int VarArgsFrameIndex;
  mutable int CachedEStackSize;
  std::vector<std::pair<MCSymbol*, CalleeSavedInfo> > SpillLabels;

public:
  XCoreFunctionInfo() :
    LRSpillSlotSet(false),
    LRSpillSlot(0),
    FPSpillSlotSet(false),
    FPSpillSlot(0),
    VarArgsFrameIndex(0),
    CachedEStackSize(-1) {}
  
  explicit XCoreFunctionInfo(MachineFunction &MF) :
    LRSpillSlotSet(false),
    LRSpillSlot(0),
    FPSpillSlotSet(false),
    FPSpillSlot(0),
    VarArgsFrameIndex(0),
    CachedEStackSize(-1) {}
  
  ~XCoreFunctionInfo() {}
  
  void setVarArgsFrameIndex(int off) { VarArgsFrameIndex = off; }
  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }

  int createLRSpillSlot(MachineFunction &MF);
  bool hasLRSpillSlot() { return LRSpillSlotSet; }
  int getLRSpillSlot() const {
    assert(LRSpillSlotSet && "LR Spill slot no set");
    return LRSpillSlot;
  }

  int createFPSpillSlot(MachineFunction &MF);
  bool hasFPSpillSlot() { return FPSpillSlotSet; }
  int getFPSpillSlot() const {
    assert(FPSpillSlotSet && "FP Spill slot no set");
    return FPSpillSlot;
  }

  bool isLargeFrame(const MachineFunction &MF) const;

  std::vector<std::pair<MCSymbol*, CalleeSavedInfo> > &getSpillLabels() {
    return SpillLabels;
  }
};
} // End llvm namespace

#endif // XCOREMACHINEFUNCTIONINFO_H
