//===-- HexagonISelLowering.h - Hexagon DAG Lowering Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Hexagon uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONISELLOWERING_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONISELLOWERING_H

#include "Hexagon.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Target/TargetLowering.h"

namespace llvm {

// Return true when the given node fits in a positive half word.
bool isPositiveHalfWord(SDNode *N);

  namespace HexagonISD {
    enum {
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      CONST32,
      CONST32_GP,  // For marking data present in GP.
      CONST32_Int_Real,
      FCONST32,
      SETCC,
      ADJDYNALLOC,
      ARGEXTEND,

      PIC_ADD,
      AT_GOT,
      AT_PCREL,

      CMPICC,      // Compare two GPR operands, set icc.
      CMPFCC,      // Compare two FP operands, set fcc.
      BRICC,       // Branch to dest on icc condition
      BRFCC,       // Branch to dest on fcc condition
      SELECT_ICC,  // Select between two values using the current ICC flags.
      SELECT_FCC,  // Select between two values using the current FCC flags.

      Hi, Lo,      // Hi/Lo operations, typically on a global address.

      FTOI,        // FP to Int within a FP register.
      ITOF,        // Int to FP within a FP register.

      CALLv3,      // A V3+ call instruction.
      CALLv3nr,    // A V3+ call instruction that doesn't return.
      CALLR,

      RET_FLAG,    // Return with a flag operand.
      BR_JT,       // Jump table.
      BARRIER,     // Memory barrier
      POPCOUNT,
      COMBINE,
      PACKHL,
      JT,
      CP,
      INSERT_ri,
      INSERT_rd,
      INSERT_riv,
      INSERT_rdv,
      EXTRACTU_ri,
      EXTRACTU_rd,
      EXTRACTU_riv,
      EXTRACTU_rdv,
      WrapperCombineII,
      WrapperCombineRR,
      WrapperCombineRI_V4,
      WrapperCombineIR_V4,
      WrapperPackhl,
      WrapperSplatB,
      WrapperSplatH,
      WrapperShuffEB,
      WrapperShuffEH,
      WrapperShuffOB,
      WrapperShuffOH,
      TC_RETURN,
      EH_RETURN,
      DCFETCH
    };
  }

  class HexagonSubtarget;

  class HexagonTargetLowering : public TargetLowering {
    int VarArgsFrameOffset;   // Frame offset to start of varargs area.

    bool CanReturnSmallStruct(const Function* CalleeFn,
                              unsigned& RetSize) const;

  public:
    const HexagonSubtarget *Subtarget;
    explicit HexagonTargetLowering(const TargetMachine &TM,
                                   const HexagonSubtarget &Subtarget);

    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization. Targets which want to do tail call
    /// optimization should implement this function.
    bool
    IsEligibleForTailCallOptimization(SDValue Callee,
                                      CallingConv::ID CalleeCC,
                                      bool isVarArg,
                                      bool isCalleeStructRet,
                                      bool isCallerStructRet,
                                      const
                                      SmallVectorImpl<ISD::OutputArg> &Outs,
                                      const SmallVectorImpl<SDValue> &OutVals,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SelectionDAG& DAG) const;

    bool isTruncateFree(Type *Ty1, Type *Ty2) const override;
    bool isTruncateFree(EVT VT1, EVT VT2) const override;

    bool allowTruncateForTailCall(Type *Ty1, Type *Ty2) const override;

    SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

    const char *getTargetNodeName(unsigned Opcode) const override;
    SDValue  LowerBR_JT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINLINEASM(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEH_LABEL(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEH_RETURN(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFormalArguments(SDValue Chain,
                                 CallingConv::ID CallConv, bool isVarArg,
                                 const SmallVectorImpl<ISD::InputArg> &Ins,
                                 SDLoc dl, SelectionDAG &DAG,
                                 SmallVectorImpl<SDValue> &InVals) const override;
    SDValue LowerGLOBALADDRESS(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                      SmallVectorImpl<SDValue> &InVals) const override;

    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            SDLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals,
                            const SmallVectorImpl<SDValue> &OutVals,
                            SDValue Callee) const;

    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerReturn(SDValue Chain,
                        CallingConv::ID CallConv, bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        const SmallVectorImpl<SDValue> &OutVals,
                        SDLoc dl, SelectionDAG &DAG) const override;

    MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr *MI,
                                MachineBasicBlock *BB) const override;

    SDValue  LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue  LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    EVT getSetCCResultType(LLVMContext &C, EVT VT) const override {
      if (!VT.isVector())
        return MVT::i1;
      else
        return EVT::getVectorVT(C, MVT::i1, VT.getVectorNumElements());
    }

    bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                    SDValue &Base, SDValue &Offset,
                                    ISD::MemIndexedMode &AM,
                                    SelectionDAG &DAG) const override;

    std::pair<unsigned, const TargetRegisterClass *>
    getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                 const std::string &Constraint,
                                 MVT VT) const override;

    unsigned getInlineAsmMemConstraint(
        const std::string &ConstraintCode) const override {
      if (ConstraintCode == "o")
        return InlineAsm::Constraint_o;
      else if (ConstraintCode == "v")
        return InlineAsm::Constraint_v;
      return TargetLowering::getInlineAsmMemConstraint(ConstraintCode);
    }

    // Intrinsics
    SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    /// The type may be VoidTy, in which case only return true if the addressing
    /// mode is legal for a load/store of any legal type.
    /// TODO: Handle pre/postinc as well.
    bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const override;
    bool isFPImmLegal(const APFloat &Imm, EVT VT) const override;

    /// isLegalICmpImmediate - Return true if the specified immediate is legal
    /// icmp immediate, that is the target has icmp instructions which can
    /// compare a register against the immediate without having to materialize
    /// the immediate into a register.
    bool isLegalICmpImmediate(int64_t Imm) const override;
  };
} // end namespace llvm

#endif    // Hexagon_ISELLOWERING_H
