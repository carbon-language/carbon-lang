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
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {
class PTXTargetMachine;

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

    // static helper routines

    static MachineSDNode *GetPTXMachineNode(SelectionDAG *DAG, unsigned Opcode,
                                            DebugLoc dl, EVT VT,
                                            SDValue Op1) {
      SDValue pred_reg = DAG->getRegister(0, MVT::i1);
      SDValue pred_imm = DAG->getTargetConstant(0, MVT::i32);
      SDValue ops[] = { Op1, pred_reg, pred_imm };
      return DAG->getMachineNode(Opcode, dl, VT, ops, array_lengthof(ops));
    }

    static MachineSDNode *GetPTXMachineNode(SelectionDAG *DAG, unsigned Opcode,
                                            DebugLoc dl, EVT VT,
                                            SDValue Op1,
                                            SDValue Op2) {
      SDValue pred_reg = DAG->getRegister(0, MVT::i1);
      SDValue pred_imm = DAG->getTargetConstant(0, MVT::i32);
      SDValue ops[] = { Op1, Op2, pred_reg, pred_imm };
      return DAG->getMachineNode(Opcode, dl, VT, ops, array_lengthof(ops));
    }

  }; // class PTXInstrInfo
} // namespace llvm

#endif // PTX_INSTR_INFO_H
