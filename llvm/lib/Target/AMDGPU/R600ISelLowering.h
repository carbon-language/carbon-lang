//===-- R600ISelLowering.h - R600 DAG Lowering Interface -*- C++ -*--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// R600 DAG Lowering interface definition
//
//===----------------------------------------------------------------------===//

#ifndef R600ISELLOWERING_H
#define R600ISELLOWERING_H

#include "AMDGPUISelLowering.h"

namespace llvm {

class R600InstrInfo;

class R600TargetLowering : public AMDGPUTargetLowering
{
public:
  R600TargetLowering(TargetMachine &TM);
  virtual MachineBasicBlock * EmitInstrWithCustomInserter(MachineInstr *MI,
      MachineBasicBlock * BB) const;
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

private:
  const R600InstrInfo * TII;

  /// lowerImplicitParameter - Each OpenCL kernel has nine implicit parameters
  /// that are stored in the first nine dwords of a Vertex Buffer.  These
  /// implicit parameters are represented by pseudo instructions, which are
  /// lowered to VTX_READ instructions by this function. 
  void lowerImplicitParameter(MachineInstr *MI, MachineBasicBlock &BB,
      MachineRegisterInfo & MRI, unsigned dword_offset) const;

  /// LowerROTL - Lower ROTL opcode to BITALIGN
  SDValue LowerROTL(SDValue Op, SelectionDAG &DAG) const;

};

} // End namespace llvm;

#endif // R600ISELLOWERING_H
