//====- ARMMachineFuctionInfo.h - ARM machine function info -----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares ARM-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef ARMMACHINEFUNCTIONINFO_H
#define ARMMACHINEFUNCTIONINFO_H

#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

/// ARMFunctionInfo - This class is derived from MachineFunction private
/// ARM target-specific information for each MachineFunction.
class ARMFunctionInfo : public MachineFunctionInfo {

  /// isThumb - True if this function is compiled under Thumb mode.
  ///
  bool isThumb;

  /// VarArgsRegSaveSize - Size of the register save area for vararg functions.
  ///
  unsigned VarArgsRegSaveSize;

  /// HasStackFrame - True if this function has a stack frame. Set by
  /// processFunctionBeforeCalleeSavedScan().
  bool HasStackFrame;

  /// LRSpilled - True if the LR register has been spilled.
  ///
  bool LRSpilled;

  /// FramePtrSpillOffset - If HasStackFrame, this records the frame pointer
  /// spill stack offset.
  unsigned FramePtrSpillOffset;

  /// GPRCS1Offset, GPRCS2Offset, DPRCSOffset - Starting offset of callee saved
  /// register spills areas. For Mac OS X:
  ///
  /// GPR callee-saved (1) : r4, r5, r6, r7, lr
  /// --------------------------------------------
  /// GPR callee-saved (2) : r8, r10, r11
  /// --------------------------------------------
  /// DPR callee-saved : d8 - d15
  unsigned GPRCS1Offset;
  unsigned GPRCS2Offset;
  unsigned DPRCSOffset;

  /// GPRCS1Size, GPRCS2Size, DPRCSSize - Sizes of callee saved register spills
  /// areas.
  unsigned GPRCS1Size;
  unsigned GPRCS2Size;
  unsigned DPRCSSize;

  /// GPRCS1Frames, GPRCS2Frames, DPRCSFrames - Keeps track of frame indices
  /// which belong to these spill areas.
  std::vector<bool> GPRCS1Frames;
  std::vector<bool> GPRCS2Frames;
  std::vector<bool> DPRCSFrames;

  /// JumpTableUId - Unique id for jumptables.
  ///
  unsigned JumpTableUId;

public:
  ARMFunctionInfo() :
    isThumb(false),
    VarArgsRegSaveSize(0), HasStackFrame(false), LRSpilled(false),
    FramePtrSpillOffset(0), GPRCS1Offset(0), GPRCS2Offset(0), DPRCSOffset(0),
    GPRCS1Size(0), GPRCS2Size(0), DPRCSSize(0), JumpTableUId(0) {}

  ARMFunctionInfo(MachineFunction &MF) :
    isThumb(MF.getTarget().getSubtarget<ARMSubtarget>().isThumb()),
    VarArgsRegSaveSize(0), HasStackFrame(false), LRSpilled(false),
    FramePtrSpillOffset(0), GPRCS1Offset(0), GPRCS2Offset(0), DPRCSOffset(0),
    GPRCS1Size(0), GPRCS2Size(0), DPRCSSize(0), JumpTableUId(0) {}

  bool isThumbFunction() const { return isThumb; }

  unsigned getVarArgsRegSaveSize() const { return VarArgsRegSaveSize; }
  void setVarArgsRegSaveSize(unsigned s) { VarArgsRegSaveSize = s; }

  bool hasStackFrame() const { return HasStackFrame; }
  void setHasStackFrame(bool s) { HasStackFrame = s; }

  bool isLRSpilled() const { return LRSpilled; }
  void setLRIsSpilled(bool s) { LRSpilled = s; }

  unsigned getFramePtrSpillOffset() const { return FramePtrSpillOffset; }
  void setFramePtrSpillOffset(unsigned o) { FramePtrSpillOffset = o; }
  
  unsigned getGPRCalleeSavedArea1Offset() const { return GPRCS1Offset; }
  unsigned getGPRCalleeSavedArea2Offset() const { return GPRCS2Offset; }
  unsigned getDPRCalleeSavedAreaOffset()  const { return DPRCSOffset; }

  void setGPRCalleeSavedArea1Offset(unsigned o) { GPRCS1Offset = o; }
  void setGPRCalleeSavedArea2Offset(unsigned o) { GPRCS2Offset = o; }
  void setDPRCalleeSavedAreaOffset(unsigned o)  { DPRCSOffset = o; }

  unsigned getGPRCalleeSavedArea1Size() const { return GPRCS1Size; }
  unsigned getGPRCalleeSavedArea2Size() const { return GPRCS2Size; }
  unsigned getDPRCalleeSavedAreaSize()  const { return DPRCSSize; }

  void setGPRCalleeSavedArea1Size(unsigned s) { GPRCS1Size = s; }
  void setGPRCalleeSavedArea2Size(unsigned s) { GPRCS2Size = s; }
  void setDPRCalleeSavedAreaSize(unsigned s)  { DPRCSSize = s; }

  bool isGPRCalleeSavedArea1Frame(int fi) const {
    if (fi < 0 || fi >= (int)GPRCS1Frames.size())
      return false;
    return GPRCS1Frames[fi];
  }
  bool isGPRCalleeSavedArea2Frame(int fi) const {
    if (fi < 0 || fi >= (int)GPRCS2Frames.size())
      return false;
    return GPRCS2Frames[fi];
  }
  bool isDPRCalleeSavedAreaFrame(int fi) const {
    if (fi < 0 || fi >= (int)DPRCSFrames.size())
      return false;
    return DPRCSFrames[fi];
  }

  void addGPRCalleeSavedArea1Frame(int fi) {
    if (fi >= 0) {
      if (fi >= (int)GPRCS1Frames.size())
        GPRCS1Frames.resize(fi+1);
      GPRCS1Frames[fi] = true;
    }
  }
  void addGPRCalleeSavedArea2Frame(int fi) {
    if (fi >= 0) {
      if (fi >= (int)GPRCS2Frames.size())
        GPRCS2Frames.resize(fi+1);
      GPRCS2Frames[fi] = true;
    }
  }
  void addDPRCalleeSavedAreaFrame(int fi) {
    if (fi >= 0) {
      if (fi >= (int)DPRCSFrames.size())
        DPRCSFrames.resize(fi+1);
      DPRCSFrames[fi] = true;
    }
  }

  unsigned createJumpTableUId() {
    return JumpTableUId++;
  }
};
} // End llvm namespace

#endif // ARMMACHINEFUNCTIONINFO_H
