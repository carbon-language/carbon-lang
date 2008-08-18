//===-- FastISel.h - Definition of the FastISel class ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the FastISel class.
//  
//===----------------------------------------------------------------------===//
  
#ifndef LLVM_CODEGEN_FASTISEL_H
#define LLVM_CODEGEN_FASTISEL_H

#include "llvm/BasicBlock.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

namespace llvm {

class MachineBasicBlock;
class MachineFunction;
class TargetInstrInfo;
class TargetRegisterClass;

/// FastISel - This is a fast-path instruction selection class that
/// generates poor code and doesn't support illegal types or non-trivial
/// lowering, but runs quickly.
class FastISel {
  MachineBasicBlock *MBB;
  MachineFunction *MF;
  const TargetInstrInfo *TII;

public:
  /// SelectInstructions - Do "fast" instruction selection over the
  /// LLVM IR instructions in the range [Begin, N) where N is either
  /// End or the first unsupported instruction. Return N.
  /// ValueMap is filled in with a mapping of LLVM IR Values to
  /// register numbers.
  BasicBlock::iterator
  SelectInstructions(BasicBlock::iterator Begin, BasicBlock::iterator End,
                     DenseMap<const Value*, unsigned> &ValueMap);

protected:
  FastISel(MachineBasicBlock *mbb, MachineFunction *mf,
           const TargetInstrInfo *tii)
    : MBB(mbb), MF(mf), TII(tii) {}

  virtual ~FastISel();

  virtual unsigned FastEmit_(MVT::SimpleValueType VT,
                             ISD::NodeType Opcode);
  virtual unsigned FastEmit_r(MVT::SimpleValueType VT,
                              ISD::NodeType Opcode, unsigned Op0);
  virtual unsigned FastEmit_rr(MVT::SimpleValueType VT,
                               ISD::NodeType Opcode,
                               unsigned Op0, unsigned Op1);

  unsigned FastEmitInst_(unsigned MachineInstOpcode,
                         const TargetRegisterClass *RC);
  unsigned FastEmitInst_r(unsigned MachineInstOpcode,
                          const TargetRegisterClass *RC,
                          unsigned Op0);
  unsigned FastEmitInst_rr(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, unsigned Op1);
};

}

#endif
