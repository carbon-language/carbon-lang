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
  bool EHSpillSlotSet;
  int EHSpillSlot[2];
  unsigned ReturnStackOffset;
  bool ReturnStackOffsetSet;
  int VarArgsFrameIndex;
  mutable int CachedEStackSize;
  std::vector<std::pair<MCSymbol*, CalleeSavedInfo> > SpillLabels;

public:
  XCoreFunctionInfo() :
    LRSpillSlotSet(false),
    FPSpillSlotSet(false),
    EHSpillSlotSet(false),
    ReturnStackOffsetSet(false),
    VarArgsFrameIndex(0),
    CachedEStackSize(-1) {}
  
  explicit XCoreFunctionInfo(MachineFunction &MF) :
    LRSpillSlotSet(false),
    FPSpillSlotSet(false),
    EHSpillSlotSet(false),
    VarArgsFrameIndex(0),
    CachedEStackSize(-1) {}
  
  ~XCoreFunctionInfo() {}
  
  void setVarArgsFrameIndex(int off) { VarArgsFrameIndex = off; }
  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }

  int createLRSpillSlot(MachineFunction &MF);
  bool hasLRSpillSlot() { return LRSpillSlotSet; }
  int getLRSpillSlot() const {
    assert(LRSpillSlotSet && "LR Spill slot not set");
    return LRSpillSlot;
  }

  int createFPSpillSlot(MachineFunction &MF);
  bool hasFPSpillSlot() { return FPSpillSlotSet; }
  int getFPSpillSlot() const {
    assert(FPSpillSlotSet && "FP Spill slot not set");
    return FPSpillSlot;
  }

  const int* createEHSpillSlot(MachineFunction &MF);
  bool hasEHSpillSlot() { return EHSpillSlotSet; }
  const int* getEHSpillSlot() const {
    assert(EHSpillSlotSet && "EH Spill slot not set");
    return EHSpillSlot;
  }

  void setReturnStackOffset(unsigned value) {
    assert(!ReturnStackOffsetSet && "Return stack offset set twice");
    ReturnStackOffset = value;
    ReturnStackOffsetSet = true;
  }

  unsigned getReturnStackOffset() const {
    assert(ReturnStackOffsetSet && "Return stack offset not set");
    return ReturnStackOffset;
  }

  bool isLargeFrame(const MachineFunction &MF) const;

  std::vector<std::pair<MCSymbol*, CalleeSavedInfo> > &getSpillLabels() {
    return SpillLabels;
  }
};
} // End llvm namespace

#endif // XCOREMACHINEFUNCTIONINFO_H
