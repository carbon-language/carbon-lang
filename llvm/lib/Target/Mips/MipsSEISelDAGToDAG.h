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

  unsigned getMSACtrlReg(const SDValue RegIdx) const;

  bool replaceUsesWithZeroReg(MachineRegisterInfo *MRI, const MachineInstr&);

  std::pair<SDNode*, SDNode*> selectMULT(SDNode *N, unsigned Opc, SDLoc dl,
                                         EVT Ty, bool HasLo, bool HasHi);

  SDNode *selectAddESubE(unsigned MOp, SDValue InFlag, SDValue CmpLHS,
                         SDLoc DL, SDNode *Node) const;

  virtual bool selectAddrRegImm(SDValue Addr, SDValue &Base,
                                SDValue &Offset) const;

  virtual bool selectAddrRegReg(SDValue Addr, SDValue &Base,
                                SDValue &Offset) const;

  virtual bool selectAddrDefault(SDValue Addr, SDValue &Base,
                                 SDValue &Offset) const;

  virtual bool selectIntAddr(SDValue Addr, SDValue &Base,
                             SDValue &Offset) const;

  virtual bool selectAddrRegImm12(SDValue Addr, SDValue &Base,
                                  SDValue &Offset) const;

  virtual bool selectIntAddrMM(SDValue Addr, SDValue &Base,
                               SDValue &Offset) const;

  /// \brief Select constant vector splats.
  virtual bool selectVSplat(SDNode *N, APInt &Imm) const;
  /// \brief Select constant vector splats whose value fits in a given integer.
  virtual bool selectVSplatCommon(SDValue N, SDValue &Imm, bool Signed,
                                  unsigned ImmBitSize) const;
  /// \brief Select constant vector splats whose value fits in a uimm1.
  virtual bool selectVSplatUimm1(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value fits in a uimm2.
  virtual bool selectVSplatUimm2(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value fits in a uimm3.
  virtual bool selectVSplatUimm3(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value fits in a uimm4.
  virtual bool selectVSplatUimm4(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value fits in a uimm5.
  virtual bool selectVSplatUimm5(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value fits in a uimm6.
  virtual bool selectVSplatUimm6(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value fits in a uimm8.
  virtual bool selectVSplatUimm8(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value fits in a simm5.
  virtual bool selectVSplatSimm5(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value is a power of 2.
  virtual bool selectVSplatUimmPow2(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value is the inverse of a
  /// power of 2.
  virtual bool selectVSplatUimmInvPow2(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value is a run of set bits
  /// ending at the most significant bit
  virtual bool selectVSplatMaskL(SDValue N, SDValue &Imm) const;
  /// \brief Select constant vector splats whose value is a run of set bits
  /// starting at bit zero.
  virtual bool selectVSplatMaskR(SDValue N, SDValue &Imm) const;

  virtual std::pair<bool, SDNode*> selectNode(SDNode *Node);

  virtual void processFunctionAfterISel(MachineFunction &MF);

  // Insert instructions to initialize the global base register in the
  // first MBB of the function.
  void initGlobalBaseReg(MachineFunction &MF);
};

FunctionPass *createMipsSEISelDag(MipsTargetMachine &TM);

}

#endif
