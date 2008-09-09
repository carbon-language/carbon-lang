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

class ConstantFP;
class MachineBasicBlock;
class MachineConstantPool;
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
  DenseMap<const Value *, unsigned> LocalValueMap;
  DenseMap<const Value *, unsigned> &ValueMap;
  DenseMap<const BasicBlock *, MachineBasicBlock *> &MBBMap;
  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  const TargetMachine &TM;
  const TargetData &TD;
  const TargetInstrInfo &TII;
  const TargetLowering &TLI;

public:
  /// setCurrentBlock - Set the current block, to which generated
  /// machine instructions will be appended.
  ///
  void setCurrentBlock(MachineBasicBlock *mbb) {
    MBB = mbb;
  }

  /// SelectInstruction - Do "fast" instruction selection for the given
  /// LLVM IR instruction, and append generated machine instructions to
  /// the current block. Return true if selection was successful.
  ///
  bool SelectInstruction(Instruction *I);

  /// SelectInstruction - Do "fast" instruction selection for the given
  /// LLVM IR operator (Instruction or ConstantExpr), and append
  /// generated machine instructions to the current block. Return true
  /// if selection was successful.
  ///
  bool SelectOperator(User *I, unsigned Opcode);

  /// TargetSelectInstruction - This method is called by target-independent
  /// code when the normal FastISel process fails to select an instruction.
  /// This gives targets a chance to emit code for anything that doesn't
  /// fit into FastISel's framework. It returns true if it was successful.
  ///
  virtual bool
  TargetSelectInstruction(Instruction *I) = 0;

  /// getRegForValue - Create a virtual register and arrange for it to
  /// be assigned the value for the given LLVM value.
  unsigned getRegForValue(Value *V);

  /// lookUpRegForValue - Look up the value to see if its value is already
  /// cached in a register. It may be defined by instructions across blocks or
  /// defined locally.
  unsigned lookUpRegForValue(Value *V);

  virtual ~FastISel();

protected:
  FastISel(MachineFunction &mf,
           DenseMap<const Value *, unsigned> &vm,
           DenseMap<const BasicBlock *, MachineBasicBlock *> &bm);

  /// FastEmit_r - This method is called by target-independent code
  /// to request that an instruction with the given type and opcode
  /// be emitted.
  virtual unsigned FastEmit_(MVT::SimpleValueType VT,
                             MVT::SimpleValueType RetVT,
                             ISD::NodeType Opcode);

  /// FastEmit_r - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register operand be emitted.
  ///
  virtual unsigned FastEmit_r(MVT::SimpleValueType VT,
                              MVT::SimpleValueType RetVT,
                              ISD::NodeType Opcode, unsigned Op0);

  /// FastEmit_rr - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register operands be emitted.
  ///
  virtual unsigned FastEmit_rr(MVT::SimpleValueType VT,
                               MVT::SimpleValueType RetVT,
                               ISD::NodeType Opcode,
                               unsigned Op0, unsigned Op1);

  /// FastEmit_ri - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register and immediate operands be emitted.
  ///
  virtual unsigned FastEmit_ri(MVT::SimpleValueType VT,
                               MVT::SimpleValueType RetVT,
                               ISD::NodeType Opcode,
                               unsigned Op0, uint64_t Imm);

  /// FastEmit_rf - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register and floating-point immediate operands be emitted.
  ///
  virtual unsigned FastEmit_rf(MVT::SimpleValueType VT,
                               MVT::SimpleValueType RetVT,
                               ISD::NodeType Opcode,
                               unsigned Op0, ConstantFP *FPImm);

  /// FastEmit_rri - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register and immediate operands be emitted.
  ///
  virtual unsigned FastEmit_rri(MVT::SimpleValueType VT,
                                MVT::SimpleValueType RetVT,
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
  
  /// FastEmit_rf_ - This method is a wrapper of FastEmit_rf. It first tries
  /// to emit an instruction with an immediate operand using FastEmit_rf.
  /// If that fails, it materializes the immediate into a register and try
  /// FastEmit_rr instead.
  unsigned FastEmit_rf_(MVT::SimpleValueType VT,
                        ISD::NodeType Opcode,
                        unsigned Op0, ConstantFP *FPImm,
                        MVT::SimpleValueType ImmType);
  
  /// FastEmit_i - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// immediate operand be emitted.
  virtual unsigned FastEmit_i(MVT::SimpleValueType VT,
                              MVT::SimpleValueType RetVT,
                              ISD::NodeType Opcode,
                              uint64_t Imm);

  /// FastEmit_f - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// floating-point immediate operand be emitted.
  virtual unsigned FastEmit_f(MVT::SimpleValueType VT,
                              MVT::SimpleValueType RetVT,
                              ISD::NodeType Opcode,
                              ConstantFP *FPImm);

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

  /// FastEmitInst_rf - Emit a MachineInstr with two register operands
  /// and a result register in the given register class.
  ///
  unsigned FastEmitInst_rf(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, ConstantFP *FPImm);

  /// FastEmitInst_rri - Emit a MachineInstr with two register operands,
  /// an immediate, and a result register in the given register class.
  ///
  unsigned FastEmitInst_rri(unsigned MachineInstOpcode,
                            const TargetRegisterClass *RC,
                            unsigned Op0, unsigned Op1, uint64_t Imm);
  
  /// FastEmitInst_i - Emit a MachineInstr with a single immediate
  /// operand, and a result register in the given register class.
  unsigned FastEmitInst_i(unsigned MachineInstrOpcode,
                          const TargetRegisterClass *RC,
                          uint64_t Imm);

  /// FastEmitInst_extractsubreg - Emit a MachineInstr for an extract_subreg
  /// from a specified index of a superregister.
  unsigned FastEmitInst_extractsubreg(unsigned Op0, uint32_t Idx);

  void UpdateValueMap(Value* I, unsigned Reg);

  unsigned createResultReg(const TargetRegisterClass *RC);
  
  /// TargetMaterializeConstant - Emit a constant in a register using 
  /// target-specific logic, such as constant pool loads.
  virtual unsigned TargetMaterializeConstant(Constant* C,
                                             MachineConstantPool* MCP) {
    return 0;
  }

private:
  bool SelectBinaryOp(User *I, ISD::NodeType ISDOpcode);

  bool SelectGetElementPtr(User *I);

  bool SelectBitCast(User *I);
  
  bool SelectCast(User *I, ISD::NodeType Opcode);
};

}

#endif
