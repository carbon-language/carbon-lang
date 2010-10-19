//====- ARMMachineFuctionInfo.h - ARM machine function info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/BitVector.h"

namespace llvm {

/// ARMFunctionInfo - This class is derived from MachineFunction private
/// ARM target-specific information for each MachineFunction.
class ARMFunctionInfo : public MachineFunctionInfo {

  /// isThumb - True if this function is compiled under Thumb mode.
  /// Used to initialized Align, so must precede it.
  bool isThumb;

  /// hasThumb2 - True if the target architecture supports Thumb2. Do not use
  /// to determine if function is compiled under Thumb mode, for that use
  /// 'isThumb'.
  bool hasThumb2;

  /// VarArgsRegSaveSize - Size of the register save area for vararg functions.
  ///
  unsigned VarArgsRegSaveSize;

  /// HasStackFrame - True if this function has a stack frame. Set by
  /// processFunctionBeforeCalleeSavedScan().
  bool HasStackFrame;

  /// RestoreSPFromFP - True if epilogue should restore SP from FP. Set by
  /// emitPrologue.
  bool RestoreSPFromFP;

  /// LRSpilledForFarJump - True if the LR register has been for spilled to
  /// enable far jump.
  bool LRSpilledForFarJump;

  /// FramePtrSpillOffset - If HasStackFrame, this records the frame pointer
  /// spill stack offset.
  unsigned FramePtrSpillOffset;

  /// GPRCSOffset, DPRCSOffset - Starting offset of callee saved register
  /// spills areas (excluding R9 for Mac OS X):
  ///
  /// GPR callee-saved (1) : r4, r5, r6, r7, r8, r9, r10, r11, lr
  /// --------------------------------------------
  /// DPR callee-saved : d8 - d15
  unsigned GPRCSOffset;
  unsigned DPRCSOffset;

  /// GPRCSSize, DPRCSSize - Sizes of callee saved register spills areas.
  unsigned GPRCSSize;
  unsigned DPRCSSize;

  /// GPRCSFrames, DPRCSFrames - Keeps track of frame indices which belong
  /// to these spill areas.
  BitVector GPRCSFrames;
  BitVector DPRCSFrames;

  /// SpilledCSRegs - A BitVector mask of all spilled callee-saved registers.
  ///
  BitVector SpilledCSRegs;

  /// JumpTableUId - Unique id for jumptables.
  ///
  unsigned JumpTableUId;

  unsigned ConstPoolEntryUId;

  /// VarArgsFrameIndex - FrameIndex for start of varargs area.
  int VarArgsFrameIndex;

  /// HasITBlocks - True if IT blocks have been inserted.
  bool HasITBlocks;

public:
  ARMFunctionInfo() :
    isThumb(false),
    hasThumb2(false),
    VarArgsRegSaveSize(0), HasStackFrame(false), RestoreSPFromFP(false),
    LRSpilledForFarJump(false),
    FramePtrSpillOffset(0), GPRCSOffset(0), DPRCSOffset(0),
    GPRCSSize(0), DPRCSSize(0),
    GPRCSFrames(0), DPRCSFrames(0),
    JumpTableUId(0), ConstPoolEntryUId(0), VarArgsFrameIndex(0),
    HasITBlocks(false) {}

  explicit ARMFunctionInfo(MachineFunction &MF) :
    isThumb(MF.getTarget().getSubtarget<ARMSubtarget>().isThumb()),
    hasThumb2(MF.getTarget().getSubtarget<ARMSubtarget>().hasThumb2()),
    VarArgsRegSaveSize(0), HasStackFrame(false), RestoreSPFromFP(false),
    LRSpilledForFarJump(false),
    FramePtrSpillOffset(0), GPRCSOffset(0), DPRCSOffset(0),
    GPRCSSize(0), DPRCSSize(0),
    GPRCSFrames(32), DPRCSFrames(32),
    SpilledCSRegs(MF.getTarget().getRegisterInfo()->getNumRegs()),
    JumpTableUId(0), ConstPoolEntryUId(0), VarArgsFrameIndex(0),
    HasITBlocks(false) {}

  bool isThumbFunction() const { return isThumb; }
  bool isThumb1OnlyFunction() const { return isThumb && !hasThumb2; }
  bool isThumb2Function() const { return isThumb && hasThumb2; }

  unsigned getVarArgsRegSaveSize() const { return VarArgsRegSaveSize; }
  void setVarArgsRegSaveSize(unsigned s) { VarArgsRegSaveSize = s; }

  bool hasStackFrame() const { return HasStackFrame; }
  void setHasStackFrame(bool s) { HasStackFrame = s; }

  bool shouldRestoreSPFromFP() const { return RestoreSPFromFP; }
  void setShouldRestoreSPFromFP(bool s) { RestoreSPFromFP = s; }

  bool isLRSpilledForFarJump() const { return LRSpilledForFarJump; }
  void setLRIsSpilledForFarJump(bool s) { LRSpilledForFarJump = s; }

  unsigned getFramePtrSpillOffset() const { return FramePtrSpillOffset; }
  void setFramePtrSpillOffset(unsigned o) { FramePtrSpillOffset = o; }

  unsigned getGPRCalleeSavedAreaOffset() const { return GPRCSOffset; }
  unsigned getDPRCalleeSavedAreaOffset()  const { return DPRCSOffset; }

  void setGPRCalleeSavedAreaOffset(unsigned o) { GPRCSOffset = o; }
  void setDPRCalleeSavedAreaOffset(unsigned o)  { DPRCSOffset = o; }

  unsigned getGPRCalleeSavedAreaSize() const { return GPRCSSize; }
  unsigned getDPRCalleeSavedAreaSize()  const { return DPRCSSize; }

  void setGPRCalleeSavedAreaSize(unsigned s) { GPRCSSize = s; }
  void setDPRCalleeSavedAreaSize(unsigned s)  { DPRCSSize = s; }

  bool isGPRCalleeSavedAreaFrame(int fi) const {
    if (fi < 0 || fi >= (int)GPRCSFrames.size())
      return false;
    return GPRCSFrames[fi];
  }
  bool isDPRCalleeSavedAreaFrame(int fi) const {
    if (fi < 0 || fi >= (int)DPRCSFrames.size())
      return false;
    return DPRCSFrames[fi];
  }

  void addGPRCalleeSavedAreaFrame(int fi) {
    if (fi >= 0) {
      int Size = GPRCSFrames.size();
      if (fi >= Size) {
        Size *= 2;
        if (fi >= Size)
          Size = fi+1;
        GPRCSFrames.resize(Size);
      }
      GPRCSFrames[fi] = true;
    }
  }
  void addDPRCalleeSavedAreaFrame(int fi) {
    if (fi >= 0) {
      int Size = DPRCSFrames.size();
      if (fi >= Size) {
        Size *= 2;
        if (fi >= Size)
          Size = fi+1;
        DPRCSFrames.resize(Size);
      }
      DPRCSFrames[fi] = true;
    }
  }

  void setCSRegisterIsSpilled(unsigned Reg) {
    SpilledCSRegs.set(Reg);
  }

  bool isCSRegisterSpilled(unsigned Reg) const {
    return SpilledCSRegs[Reg];
  }

  const BitVector &getSpilledCSRegisters() const {
    return SpilledCSRegs;
  }

  unsigned createJumpTableUId() {
    return JumpTableUId++;
  }

  unsigned getNumJumpTables() const {
    return JumpTableUId;
  }

  void initConstPoolEntryUId(unsigned UId) {
    ConstPoolEntryUId = UId;
  }

  unsigned getNumConstPoolEntries() const {
    return ConstPoolEntryUId;
  }

  unsigned createConstPoolEntryUId() {
    return ConstPoolEntryUId++;
  }

  int getVarArgsFrameIndex() const { return VarArgsFrameIndex; }
  void setVarArgsFrameIndex(int Index) { VarArgsFrameIndex = Index; }

  bool hasITBlocks() const { return HasITBlocks; }
  void setHasITBlocks(bool h) { HasITBlocks = h; }
};
} // End llvm namespace

#endif // ARMMACHINEFUNCTIONINFO_H
