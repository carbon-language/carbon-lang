//===-- MipsSEISelDAGToDAG.h - A Dag to Dag Inst Selector for MipsSE -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Subclass of MipsDAGToDAGISel specialized for mips32/64.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSSEISELDAGTODAG_H
#define MIPSSEISELDAGTODAG_H

#include "MipsISelDAGToDAG.h"

namespace llvm {

class MipsSEDAGToDAGISel : public MipsDAGToDAGISel {

public:
  explicit MipsSEDAGToDAGISel(MipsTargetMachine &TM) : MipsDAGToDAGISel(TM) {}

private:

  virtual bool runOnMachineFunction(MachineFunction &MF);

  void addDSPCtrlRegOperands(bool IsDef, MachineInstr &MI,
                             MachineFunction &MF);

  bool replaceUsesWithZeroReg(MachineRegisterInfo *MRI, const MachineInstr&);

  std::pair<SDNode*, SDNode*> selectMULT(SDNode *N, unsigned Opc, SDLoc dl,
                                         EVT Ty, bool HasLo, bool HasHi);

  SDNode *selectAddESubE(unsigned MOp, SDValue InFlag, SDValue CmpLHS,
                         SDLoc DL, SDNode *Node) const;

  virtual bool selectAddrRegImm(SDValue Addr, SDValue &Base,
                                SDValue &Offset) const;

  virtual bool selectAddrDefault(SDValue Addr, SDValue &Base,
                                 SDValue &Offset) const;

  virtual bool selectIntAddr(SDValue Addr, SDValue &Base,
                             SDValue &Offset) const;

  virtual std::pair<bool, SDNode*> selectNode(SDNode *Node);

  virtual void processFunctionAfterISel(MachineFunction &MF);

  // Insert instructions to initialize the global base register in the
  // first MBB of the function.
  void initGlobalBaseReg(MachineFunction &MF);
};

FunctionPass *createMipsSEISelDag(MipsTargetMachine &TM);

}

#endif
