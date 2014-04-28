//===- HexagonInstrInfo.h - Hexagon Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonINSTRUCTIONINFO_H
#define HexagonINSTRUCTIONINFO_H

#include "HexagonRegisterInfo.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "HexagonGenInstrInfo.inc"

namespace llvm {

struct EVT;

class HexagonInstrInfo : public HexagonGenInstrInfo {
  virtual void anchor();
  const HexagonRegisterInfo RI;
  const HexagonSubtarget &Subtarget;
  typedef unsigned Opcode_t;

public:
  explicit HexagonInstrInfo(HexagonSubtarget &ST);

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const HexagonRegisterInfo &getRegisterInfo() const { return RI; }

  /// isLoadFromStackSlot - If the specified machine instruction is a direct
  /// load from a stack slot, return the virtual or physical register number of
  /// the destination along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than loading from the stack slot.
  virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                       int &FrameIndex) const;

  /// isStoreToStackSlot - If the specified machine instruction is a direct
  /// store to a stack slot, return the virtual or physical register number of
  /// the source reg along with the FrameIndex of the loaded stack slot.  If
  /// not, return 0.  This predicate must return 0 if the instruction has
  /// any side effects other than storing to the stack slot.
  virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                      int &FrameIndex) const;


  virtual bool AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const;

  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;

  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const;

  virtual bool analyzeCompare(const MachineInstr *MI,
                              unsigned &SrcReg, unsigned &SrcReg2,
                              int &Mask, int &Value) const;

  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           unsigned DestReg, unsigned SrcReg,
                           bool KillSrc) const;

  virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const;

  virtual void storeRegToAddr(MachineFunction &MF, unsigned SrcReg, bool isKill,
                              SmallVectorImpl<MachineOperand> &Addr,
                              const TargetRegisterClass *RC,
                              SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    unsigned DestReg, int FrameIndex,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const;

  virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                               SmallVectorImpl<MachineOperand> &Addr,
                               const TargetRegisterClass *RC,
                               SmallVectorImpl<MachineInstr*> &NewMIs) const;

  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                              int FrameIndex) const;

  virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
                                              MachineInstr* MI,
                                           const SmallVectorImpl<unsigned> &Ops,
                                              MachineInstr* LoadMI) const {
    return nullptr;
  }

  unsigned createVR(MachineFunction* MF, MVT VT) const;

  virtual bool isBranch(const MachineInstr *MI) const;
  virtual bool isPredicable(MachineInstr *MI) const;
  virtual bool
  PredicateInstruction(MachineInstr *MI,
                       const SmallVectorImpl<MachineOperand> &Cond) const;

  virtual bool isProfitableToIfCvt(MachineBasicBlock &MBB, unsigned NumCycles,
                                   unsigned ExtraPredCycles,
                                   const BranchProbability &Probability) const;

  virtual bool isProfitableToIfCvt(MachineBasicBlock &TMBB,
                                   unsigned NumTCycles, unsigned ExtraTCycles,
                                   MachineBasicBlock &FMBB,
                                   unsigned NumFCycles, unsigned ExtraFCycles,
                                   const BranchProbability &Probability) const;

  virtual bool isPredicated(const MachineInstr *MI) const;
  virtual bool isPredicated(unsigned Opcode) const;
  virtual bool isPredicatedTrue(const MachineInstr *MI) const;
  virtual bool isPredicatedTrue(unsigned Opcode) const;
  virtual bool isPredicatedNew(const MachineInstr *MI) const;
  virtual bool isPredicatedNew(unsigned Opcode) const;
  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const;
  virtual bool
  SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                    const SmallVectorImpl<MachineOperand> &Pred2) const;

  virtual bool
  ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const;

  virtual bool
  isProfitableToDupForIfCvt(MachineBasicBlock &MBB,unsigned NumCycles,
                            const BranchProbability &Probability) const;

  virtual DFAPacketizer*
  CreateTargetScheduleState(const TargetMachine *TM,
                            const ScheduleDAG *DAG) const;

  virtual bool isSchedulingBoundary(const MachineInstr *MI,
                                    const MachineBasicBlock *MBB,
                                    const MachineFunction &MF) const;
  bool isValidOffset(const int Opcode, const int Offset) const;
  bool isValidAutoIncImm(const EVT VT, const int Offset) const;
  bool isMemOp(const MachineInstr *MI) const;
  bool isSpillPredRegOp(const MachineInstr *MI) const;
  bool isU6_3Immediate(const int value) const;
  bool isU6_2Immediate(const int value) const;
  bool isU6_1Immediate(const int value) const;
  bool isU6_0Immediate(const int value) const;
  bool isS4_3Immediate(const int value) const;
  bool isS4_2Immediate(const int value) const;
  bool isS4_1Immediate(const int value) const;
  bool isS4_0Immediate(const int value) const;
  bool isS12_Immediate(const int value) const;
  bool isU6_Immediate(const int value) const;
  bool isS8_Immediate(const int value) const;
  bool isS6_Immediate(const int value) const;

  bool isSaveCalleeSavedRegsCall(const MachineInstr* MI) const;
  bool isConditionalTransfer(const MachineInstr* MI) const;
  bool isConditionalALU32 (const MachineInstr* MI) const;
  bool isConditionalLoad (const MachineInstr* MI) const;
  bool isConditionalStore(const MachineInstr* MI) const;
  bool isNewValueInst(const MachineInstr* MI) const;
  bool isNewValue(const MachineInstr* MI) const;
  bool isDotNewInst(const MachineInstr* MI) const;
  int GetDotOldOp(const int opc) const;
  int GetDotNewOp(const MachineInstr* MI) const;
  int GetDotNewPredOp(MachineInstr *MI,
                      const MachineBranchProbabilityInfo
                      *MBPI) const;
  bool mayBeNewStore(const MachineInstr* MI) const;
  bool isDeallocRet(const MachineInstr *MI) const;
  unsigned getInvertedPredicatedOpcode(const int Opc) const;
  bool isExtendable(const MachineInstr* MI) const;
  bool isExtended(const MachineInstr* MI) const;
  bool isPostIncrement(const MachineInstr* MI) const;
  bool isNewValueStore(const MachineInstr* MI) const;
  bool isNewValueStore(unsigned Opcode) const;
  bool isNewValueJump(const MachineInstr* MI) const;
  bool isNewValueJumpCandidate(const MachineInstr *MI) const;


  void immediateExtend(MachineInstr *MI) const;
  bool isConstExtended(MachineInstr *MI) const;
  int getDotNewPredJumpOp(MachineInstr *MI,
                      const MachineBranchProbabilityInfo *MBPI) const;
  unsigned getAddrMode(const MachineInstr* MI) const;
  bool isOperandExtended(const MachineInstr *MI,
                         unsigned short OperandNum) const;
  unsigned short getCExtOpNum(const MachineInstr *MI) const;
  int getMinValue(const MachineInstr *MI) const;
  int getMaxValue(const MachineInstr *MI) const;
  bool NonExtEquivalentExists (const MachineInstr *MI) const;
  short getNonExtOpcode(const MachineInstr *MI) const;
  bool PredOpcodeHasJMP_c(Opcode_t Opcode) const;
  bool PredOpcodeHasNot(Opcode_t Opcode) const;

private:
  int getMatchingCondBranchOpcode(int Opc, bool sense) const;

};

}

#endif
