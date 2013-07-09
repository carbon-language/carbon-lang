//===-- SystemZISelLowering.h - SystemZ DAG lowering interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that SystemZ uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_SystemZ_ISELLOWERING_H
#define LLVM_TARGET_SystemZ_ISELLOWERING_H

#include "SystemZ.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {
namespace SystemZISD {
  enum {
    FIRST_NUMBER = ISD::BUILTIN_OP_END,

    // Return with a flag operand.  Operand 0 is the chain operand.
    RET_FLAG,

    // Calls a function.  Operand 0 is the chain operand and operand 1
    // is the target address.  The arguments start at operand 2.
    // There is an optional glue operand at the end.
    CALL,

    // Wraps a TargetGlobalAddress that should be loaded using PC-relative
    // accesses (LARL).  Operand 0 is the address.
    PCREL_WRAPPER,

    // Signed integer and floating-point comparisons.  The operands are the
    // two values to compare.
    CMP,

    // Likewise unsigned integer comparison.
    UCMP,

    // Branches if a condition is true.  Operand 0 is the chain operand;
    // operand 1 is the 4-bit condition-code mask, with bit N in
    // big-endian order meaning "branch if CC=N"; operand 2 is the
    // target block and operand 3 is the flag operand.
    BR_CCMASK,

    // Selects between operand 0 and operand 1.  Operand 2 is the
    // mask of condition-code values for which operand 0 should be
    // chosen over operand 1; it has the same form as BR_CCMASK.
    // Operand 3 is the flag operand.
    SELECT_CCMASK,

    // Evaluates to the gap between the stack pointer and the
    // base of the dynamically-allocatable area.
    ADJDYNALLOC,

    // Extracts the value of a 32-bit access register.  Operand 0 is
    // the number of the register.
    EXTRACT_ACCESS,

    // Wrappers around the ISD opcodes of the same name.  The output and
    // first input operands are GR128s.  The trailing numbers are the
    // widths of the second operand in bits.
    UMUL_LOHI64,
    SDIVREM32,
    SDIVREM64,
    UDIVREM32,
    UDIVREM64,

    // Use MVC to copy bytes from one memory location to another.
    // The first operand is the target address, the second operand is the
    // source address, and the third operand is the constant length.
    // This isn't a memory opcode because we'd need to attach two
    // MachineMemOperands rather than one.
    MVC,

    // Wrappers around the inner loop of an 8- or 16-bit ATOMIC_SWAP or
    // ATOMIC_LOAD_<op>.
    //
    // Operand 0: the address of the containing 32-bit-aligned field
    // Operand 1: the second operand of <op>, in the high bits of an i32
    //            for everything except ATOMIC_SWAPW
    // Operand 2: how many bits to rotate the i32 left to bring the first
    //            operand into the high bits
    // Operand 3: the negative of operand 2, for rotating the other way
    // Operand 4: the width of the field in bits (8 or 16)
    ATOMIC_SWAPW = ISD::FIRST_TARGET_MEMORY_OPCODE,
    ATOMIC_LOADW_ADD,
    ATOMIC_LOADW_SUB,
    ATOMIC_LOADW_AND,
    ATOMIC_LOADW_OR,
    ATOMIC_LOADW_XOR,
    ATOMIC_LOADW_NAND,
    ATOMIC_LOADW_MIN,
    ATOMIC_LOADW_MAX,
    ATOMIC_LOADW_UMIN,
    ATOMIC_LOADW_UMAX,

    // A wrapper around the inner loop of an ATOMIC_CMP_SWAP.
    //
    // Operand 0: the address of the containing 32-bit-aligned field
    // Operand 1: the compare value, in the low bits of an i32
    // Operand 2: the swap value, in the low bits of an i32
    // Operand 3: how many bits to rotate the i32 left to bring the first
    //            operand into the high bits
    // Operand 4: the negative of operand 2, for rotating the other way
    // Operand 5: the width of the field in bits (8 or 16)
    ATOMIC_CMP_SWAPW
  };
}

class SystemZSubtarget;
class SystemZTargetMachine;

class SystemZTargetLowering : public TargetLowering {
public:
  explicit SystemZTargetLowering(SystemZTargetMachine &TM);

  // Override TargetLowering.
  virtual MVT getScalarShiftAmountTy(EVT LHSTy) const LLVM_OVERRIDE {
    return MVT::i32;
  }
  virtual EVT getSetCCResultType(LLVMContext &, EVT) const {
    return MVT::i32;
  }
  virtual bool isFMAFasterThanFMulAndFAdd(EVT VT) const LLVM_OVERRIDE;
  virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;
  virtual bool allowsUnalignedMemoryAccesses(EVT VT, bool *Fast) const;
  virtual const char *getTargetNodeName(unsigned Opcode) const LLVM_OVERRIDE;
  virtual std::pair<unsigned, const TargetRegisterClass *>
    getRegForInlineAsmConstraint(const std::string &Constraint,
                                 MVT VT) const LLVM_OVERRIDE;
  virtual TargetLowering::ConstraintType
    getConstraintType(const std::string &Constraint) const LLVM_OVERRIDE;
  virtual TargetLowering::ConstraintWeight
    getSingleConstraintMatchWeight(AsmOperandInfo &info,
                                   const char *constraint) const LLVM_OVERRIDE;
  virtual void
    LowerAsmOperandForConstraint(SDValue Op,
                                 std::string &Constraint,
                                 std::vector<SDValue> &Ops,
                                 SelectionDAG &DAG) const LLVM_OVERRIDE;
  virtual MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr *MI,
                                MachineBasicBlock *BB) const LLVM_OVERRIDE;
  virtual SDValue LowerOperation(SDValue Op,
                                 SelectionDAG &DAG) const LLVM_OVERRIDE;
  virtual SDValue
    LowerFormalArguments(SDValue Chain,
                         CallingConv::ID CallConv, bool isVarArg,
                         const SmallVectorImpl<ISD::InputArg> &Ins,
                         SDLoc DL, SelectionDAG &DAG,
                         SmallVectorImpl<SDValue> &InVals) const LLVM_OVERRIDE;
  virtual SDValue
    LowerCall(CallLoweringInfo &CLI,
              SmallVectorImpl<SDValue> &InVals) const LLVM_OVERRIDE;

  virtual SDValue
    LowerReturn(SDValue Chain,
                CallingConv::ID CallConv, bool IsVarArg,
                const SmallVectorImpl<ISD::OutputArg> &Outs,
                const SmallVectorImpl<SDValue> &OutVals,
                SDLoc DL, SelectionDAG &DAG) const LLVM_OVERRIDE;

private:
  const SystemZSubtarget &Subtarget;
  const SystemZTargetMachine &TM;

  // Implement LowerOperation for individual opcodes.
  SDValue lowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalAddress(GlobalAddressSDNode *Node,
                             SelectionDAG &DAG) const;
  SDValue lowerGlobalTLSAddress(GlobalAddressSDNode *Node,
                                SelectionDAG &DAG) const;
  SDValue lowerBlockAddress(BlockAddressSDNode *Node,
                            SelectionDAG &DAG) const;
  SDValue lowerJumpTable(JumpTableSDNode *JT, SelectionDAG &DAG) const;
  SDValue lowerConstantPool(ConstantPoolSDNode *CP, SelectionDAG &DAG) const;
  SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVACOPY(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerUMUL_LOHI(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerUDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerBITCAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerATOMIC_LOAD(SDValue Op, SelectionDAG &DAG,
                           unsigned Opcode) const;
  SDValue lowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSTACKSAVE(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerSTACKRESTORE(SDValue Op, SelectionDAG &DAG) const;

  // If the last instruction before MBBI in MBB was some form of COMPARE,
  // try to replace it with a COMPARE AND BRANCH just before MBBI.
  // CCMask and Target are the BRC-like operands for the branch.
  // Return true if the change was made.
  bool convertPrevCompareToBranch(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator MBBI,
                                  unsigned CCMask,
                                  MachineBasicBlock *Target) const;

  // Implement EmitInstrWithCustomInserter for individual operation types.
  MachineBasicBlock *emitSelect(MachineInstr *MI,
                                MachineBasicBlock *BB) const;
  MachineBasicBlock *emitCondStore(MachineInstr *MI,
                                   MachineBasicBlock *BB,
                                   unsigned StoreOpcode, bool Invert) const;
  MachineBasicBlock *emitExt128(MachineInstr *MI,
                                MachineBasicBlock *MBB,
                                bool ClearEven, unsigned SubReg) const;
  MachineBasicBlock *emitAtomicLoadBinary(MachineInstr *MI,
                                          MachineBasicBlock *BB,
                                          unsigned BinOpcode, unsigned BitSize,
                                          bool Invert = false) const;
  MachineBasicBlock *emitAtomicLoadMinMax(MachineInstr *MI,
                                          MachineBasicBlock *MBB,
                                          unsigned CompareOpcode,
                                          unsigned KeepOldMask,
                                          unsigned BitSize) const;
  MachineBasicBlock *emitAtomicCmpSwapW(MachineInstr *MI,
                                        MachineBasicBlock *BB) const;
  MachineBasicBlock *emitMVCWrapper(MachineInstr *MI,
                                    MachineBasicBlock *BB) const;
};
} // end namespace llvm

#endif // LLVM_TARGET_SystemZ_ISELLOWERING_H
