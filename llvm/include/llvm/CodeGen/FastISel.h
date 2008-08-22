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
#include <map>

namespace llvm {

class MachineBasicBlock;
class MachineFunction;
class MachineRegisterInfo;
class TargetData;
class TargetInstrInfo;
class TargetLowering;
class TargetMachine;
class TargetRegisterClass;

/// FastISel - This is a fast-path instruction selection class that
/// generates poor code and doesn't support illegal types or non-trivial
/// lowering, but runs quickly.
class FastISel {
protected:
  MachineBasicBlock *MBB;
  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  const TargetMachine &TM;
  const TargetData &TD;
  const TargetInstrInfo &TII;
  const TargetLowering &TLI;

public:
  /// SelectInstructions - Do "fast" instruction selection over the
  /// LLVM IR instructions in the range [Begin, N) where N is either
  /// End or the first unsupported instruction. Return N.
  /// ValueMap is filled in with a mapping of LLVM IR Values to
  /// virtual register numbers. MBB is a block to which to append
  /// the generated MachineInstrs.
  BasicBlock::iterator
  SelectInstructions(BasicBlock::iterator Begin, BasicBlock::iterator End,
                     DenseMap<const Value*, unsigned> &ValueMap,
                     std::map<const BasicBlock*, MachineBasicBlock *> &MBBMap,
                     MachineBasicBlock *MBB);

  virtual ~FastISel();

protected:
  explicit FastISel(MachineFunction &mf);

  /// FastEmit_r - This method is called by target-independent code
  /// to request that an instruction with the given type and opcode
  /// be emitted.
  virtual unsigned FastEmit_(MVT::SimpleValueType VT,
                             ISD::NodeType Opcode);

  /// FastEmit_r - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register operand be emitted.
  ///
  virtual unsigned FastEmit_r(MVT::SimpleValueType VT,
                              ISD::NodeType Opcode, unsigned Op0);

  /// FastEmit_rr - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register operands be emitted.
  ///
  virtual unsigned FastEmit_rr(MVT::SimpleValueType VT,
                               ISD::NodeType Opcode,
                               unsigned Op0, unsigned Op1);

  /// FastEmit_i - This method is called by target-independent code
  /// to request that an instruction with the given type which materialize
  /// the specified immediate value.
  virtual unsigned FastEmit_i(MVT::SimpleValueType VT, uint64_t Imm);

  /// FastEmit_ri - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register and immediate operands be emitted.
  ///
  virtual unsigned FastEmit_ri(MVT::SimpleValueType VT,
                               ISD::NodeType Opcode,
                               unsigned Op0, uint64_t Imm);

  /// FastEmit_rri - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register and immediate operands be emitted.
  ///
  virtual unsigned FastEmit_rri(MVT::SimpleValueType VT,
                                ISD::NodeType Opcode,
                                unsigned Op0, unsigned Op1, uint64_t Imm);

  /// FastEmit_ri_ - This method is a wrapper of FastEmit_ri. It first tries
  /// to emit an instruction with an immediate operand using FastEmit_ri.
  /// If that fails, it materializes the immediate into a register and try
  /// FastEmit_rr instead.
  unsigned FastEmit_ri_(MVT::SimpleValueType VT,
                        ISD::NodeType Opcode,
                        unsigned Op0, uint64_t Imm,
                        MVT::SimpleValueType ImmType);

  /// FastEmitInst_ - Emit a MachineInstr with no operands and a
  /// result register in the given register class.
  ///
  unsigned FastEmitInst_(unsigned MachineInstOpcode,
                         const TargetRegisterClass *RC);

  /// FastEmitInst_r - Emit a MachineInstr with one register operand
  /// and a result register in the given register class.
  ///
  unsigned FastEmitInst_r(unsigned MachineInstOpcode,
                          const TargetRegisterClass *RC,
                          unsigned Op0);

  /// FastEmitInst_rr - Emit a MachineInstr with two register operands
  /// and a result register in the given register class.
  ///
  unsigned FastEmitInst_rr(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, unsigned Op1);

  /// FastEmitInst_ri - Emit a MachineInstr with two register operands
  /// and a result register in the given register class.
  ///
  unsigned FastEmitInst_ri(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, uint64_t Imm);

  /// FastEmitInst_rri - Emit a MachineInstr with two register operands,
  /// an immediate, and a result register in the given register class.
  ///
  unsigned FastEmitInst_rri(unsigned MachineInstOpcode,
                            const TargetRegisterClass *RC,
                            unsigned Op0, unsigned Op1, uint64_t Imm);

private:
  unsigned createResultReg(const TargetRegisterClass *RC);

  bool SelectBinaryOp(Instruction *I, ISD::NodeType ISDOpcode,
                      DenseMap<const Value*, unsigned> &ValueMap);

  bool SelectGetElementPtr(Instruction *I,
                           DenseMap<const Value*, unsigned> &ValueMap);
};

}

#endif
