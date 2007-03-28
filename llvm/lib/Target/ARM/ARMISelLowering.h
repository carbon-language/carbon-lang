//===-- ARMISelLowering.h - ARM DAG Lowering Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that ARM uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef ARMISELLOWERING_H
#define ARMISELLOWERING_H

#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include <vector>

namespace llvm {
  class ARMConstantPoolValue;
  class ARMSubtarget;

  namespace ARMISD {
    // ARM Specific DAG Nodes
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+ARM::INSTRUCTION_LIST_END,

      Wrapper,      // Wrapper - A wrapper node for TargetConstantPool,
                    // TargetExternalSymbol, and TargetGlobalAddress.
      WrapperJT,    // WrapperJT - A wrapper node for TargetJumpTable
      
      CALL,         // Function call.
      CALL_NOLINK,  // Function call with branch not branch-and-link.
      tCALL,        // Thumb function call.
      BRCOND,       // Conditional branch.
      BR_JT,        // Jumptable branch.
      RET_FLAG,     // Return with a flag operand.

      PIC_ADD,      // Add with a PC operand and a PIC label.

      CMP,          // ARM compare instructions.
      CMPFP,        // ARM VFP compare instruction, sets FPSCR.
      CMPFPw0,      // ARM VFP compare against zero instruction, sets FPSCR.
      FMSTAT,       // ARM fmstat instruction.
      CMOV,         // ARM conditional move instructions.
      CNEG,         // ARM conditional negate instructions.
      
      FTOSI,        // FP to sint within a FP register.
      FTOUI,        // FP to uint within a FP register.
      SITOF,        // sint to FP within a FP register.
      UITOF,        // uint to FP within a FP register.

      MULHILOU,     // Lo,Hi = umul LHS, RHS.
      MULHILOS,     // Lo,Hi = smul LHS, RHS.
      
      SRL_FLAG,     // V,Flag = srl_flag X -> srl X, 1 + save carry out.
      SRA_FLAG,     // V,Flag = sra_flag X -> sra X, 1 + save carry out.
      RRX,          // V = RRX X, Flag     -> srl X, 1 + shift in carry flag.
      
      FMRRD,        // double to two gprs.
      FMDRR         // Two gprs to double.
    };
  }

  //===----------------------------------------------------------------------===//
  //  ARMTargetLowering - ARM Implementation of the TargetLowering interface
  
  class ARMTargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
  public:
    ARMTargetLowering(TargetMachine &TM);

    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB);

    /// isLegalAddressImmediate - Return true if the integer value can be used
    /// as the offset of the target addressing mode for load / store of the
    /// given type.
    virtual bool isLegalAddressImmediate(int64_t V, const Type *Ty) const;

    /// isLegalAddressImmediate - Return true if the GlobalValue can be used as
    /// the offset of the target addressing mode.
    virtual bool isLegalAddressImmediate(GlobalValue *GV) const;

    /// isLegalAddressScale - Return true if the integer value can be used as
    /// the scale of the target addressing mode for load / store of the given
    /// type.
    virtual bool isLegalAddressScale(int64_t S, const Type *Ty) const;

    /// isLegalAddressScaleAndImm - Return true if S works for 
    /// IsLegalAddressScale and V works for isLegalAddressImmediate _and_ 
    /// both can be applied simultaneously to the same instruction.
    virtual bool isLegalAddressScaleAndImm(int64_t S, int64_t V, 
                                           const Type *Ty) const;

    /// isLegalAddressScaleAndImm - Return true if S works for 
    /// IsLegalAddressScale and GV works for isLegalAddressImmediate _and_
    /// both can be applied simultaneously to the same instruction.
    virtual bool isLegalAddressScaleAndImm(int64_t S, GlobalValue *GV,
                                           const Type *Ty) const;

    /// getPreIndexedAddressParts - returns true by value, base pointer and
    /// offset pointer and addressing mode by reference if the node's address
    /// can be legally represented as pre-indexed load / store address.
    virtual bool getPreIndexedAddressParts(SDNode *N, SDOperand &Base,
                                           SDOperand &Offset,
                                           ISD::MemIndexedMode &AM,
                                           SelectionDAG &DAG);

    /// getPostIndexedAddressParts - returns true by value, base pointer and
    /// offset pointer and addressing mode by reference if this node can be
    /// combined with a load / store to form a post-indexed load / store.
    virtual bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                            SDOperand &Base, SDOperand &Offset,
                                            ISD::MemIndexedMode &AM,
                                            SelectionDAG &DAG);

    virtual void computeMaskedBitsForTargetNode(const SDOperand Op,
                                                uint64_t Mask,
                                                uint64_t &KnownZero, 
                                                uint64_t &KnownOne,
                                                unsigned Depth) const;
    ConstraintType getConstraintType(const std::string &Constraint) const;
    std::pair<unsigned, const TargetRegisterClass*> 
      getRegForInlineAsmConstraint(const std::string &Constraint,
                                   MVT::ValueType VT) const;
    std::vector<unsigned>
    getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                      MVT::ValueType VT) const;
  private:
    /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
    /// make the right decision when generating code for different targets.
    const ARMSubtarget *Subtarget;

    /// ARMPCLabelIndex - Keep track the number of ARM PC labels created.
    ///
    unsigned ARMPCLabelIndex;

    SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerBR_JT(SDOperand Op, SelectionDAG &DAG);
  };
}

#endif  // ARMISELLOWERING_H
