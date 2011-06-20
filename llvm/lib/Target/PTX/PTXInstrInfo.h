//===- PTXInstrInfo.h - PTX Instruction Information -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_INSTR_INFO_H
#define PTX_INSTR_INFO_H

#include "PTXRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {
class PTXTargetMachine;

class MachineSDNode;
class SDValue;
class SelectionDAG;

class PTXInstrInfo : public TargetInstrInfoImpl {
private:
  const PTXRegisterInfo RI;
  PTXTargetMachine &TM;

public:
  explicit PTXInstrInfo(PTXTargetMachine &_TM);

  virtual const PTXRegisterInfo &getRegisterInfo() const { return RI; }

  virtual void copyPhysReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, DebugLoc DL,
                           unsigned DstReg, unsigned SrcReg,
                           bool KillSrc) const;

  virtual bool copyRegToReg(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator I,
                            unsigned DstReg, unsigned SrcReg,
                            const TargetRegisterClass *DstRC,
                            const TargetRegisterClass *SrcRC,
                            DebugLoc DL) const;

  virtual bool isMoveInstr(const MachineInstr& MI,
                           unsigned &SrcReg, unsigned &DstReg,
                           unsigned &SrcSubIdx, unsigned &DstSubIdx) const;

  // predicate support

  virtual bool isPredicated(const MachineInstr *MI) const;

  virtual bool isUnpredicatedTerminator(const MachineInstr *MI) const;

  virtual
  bool PredicateInstruction(MachineInstr *MI,
                            const SmallVectorImpl<MachineOperand> &Pred) const;

  virtual
  bool SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                         const SmallVectorImpl<MachineOperand> &Pred2) const;

  virtual bool DefinesPredicate(MachineInstr *MI,
                                std::vector<MachineOperand> &Pred) const;

  // PTX is fully-predicable
  virtual bool isPredicable(MachineInstr *MI) const { return true; }

  // branch support

  virtual bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                             MachineBasicBlock *&FBB,
                             SmallVectorImpl<MachineOperand> &Cond,
                             bool AllowModify = false) const;

  virtual unsigned RemoveBranch(MachineBasicBlock &MBB) const;

  virtual unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const SmallVectorImpl<MachineOperand> &Cond,
                                DebugLoc DL) const;

  // Memory operand folding for spills
  // TODO: Implement this eventually and get rid of storeRegToStackSlot and
  //       loadRegFromStackSlot.  Doing so will get rid of the "stack" registers
  //       we currently use to spill, though I doubt the overall effect on ptxas
  //       output will be large.  I have yet to see a case where ptxas is unable
  //       to see through the "stack" register usage and hence generates
  //       efficient code anyway.
  // virtual MachineInstr* foldMemoryOperandImpl(MachineFunction &MF,
  //                                             MachineInstr* MI,
  //                                          const SmallVectorImpl<unsigned> &Ops,
  //                                             int FrameIndex) const;

  virtual void storeRegToStackSlot(MachineBasicBlock& MBB,
                                   MachineBasicBlock::iterator MII,
                                   unsigned SrcReg, bool isKill, int FrameIndex,
                                   const TargetRegisterClass* RC,
                                   const TargetRegisterInfo* TRI) const;
  virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MII,
                                    unsigned DestReg, int FrameIdx,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const;

  // static helper routines

  static MachineSDNode *GetPTXMachineNode(SelectionDAG *DAG, unsigned Opcode,
                                          DebugLoc dl, EVT VT,
                                          SDValue Op1);

  static MachineSDNode *GetPTXMachineNode(SelectionDAG *DAG, unsigned Opcode,
                                          DebugLoc dl, EVT VT,
                                          SDValue Op1, SDValue Op2);

  static void AddDefaultPredicate(MachineInstr *MI);

  static bool IsAnyKindOfBranch(const MachineInstr& inst);

  static bool IsAnySuccessorAlsoLayoutSuccessor(const MachineBasicBlock& MBB);

  static MachineBasicBlock *GetBranchTarget(const MachineInstr& inst);
}; // class PTXInstrInfo
} // namespace llvm

#endif // PTX_INSTR_INFO_H
