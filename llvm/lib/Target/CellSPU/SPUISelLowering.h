//===-- SPUISelLowering.h - Cell SPU DAG Lowering Interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Cell SPU uses to lower LLVM code into
// a selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_ISELLOWERING_H
#define SPU_ISELLOWERING_H

#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "SPU.h"

namespace llvm {
  namespace SPUISD {
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,
      
      // Pseudo instructions:
      RET_FLAG,                 ///< Return with flag, matched by bi instruction
      
      Hi,                       ///< High address component (upper 16)
      Lo,                       ///< Low address component (lower 16)
      PCRelAddr,                ///< Program counter relative address
      AFormAddr,                ///< A-form address (local store)
      IndirectAddr,             ///< D-Form "imm($r)" and X-form "$r($r)"

      LDRESULT,                 ///< Load result (value, chain)
      CALL,                     ///< CALL instruction
      SHUFB,                    ///< Vector shuffle (permute)
      INSERT_MASK,              ///< Insert element shuffle mask
      CNTB,                     ///< Count leading ones in bytes
      PROMOTE_SCALAR,           ///< Promote scalar->vector
      EXTRACT_ELT0,             ///< Extract element 0
      EXTRACT_ELT0_CHAINED,     ///< Extract element 0, with chain
      EXTRACT_I1_ZEXT,          ///< Extract element 0 as i1, zero extend
      EXTRACT_I1_SEXT,          ///< Extract element 0 as i1, sign extend
      EXTRACT_I8_ZEXT,          ///< Extract element 0 as i8, zero extend
      EXTRACT_I8_SEXT,          ///< Extract element 0 as i8, sign extend
      MPY,                      ///< 16-bit Multiply (low parts of a 32-bit)
      MPYU,                     ///< Multiply Unsigned
      MPYH,                     ///< Multiply High
      MPYHH,                    ///< Multiply High-High
      SHLQUAD_L_BITS,           ///< Rotate quad left, by bits
      SHLQUAD_L_BYTES,          ///< Rotate quad left, by bytes
      VEC_SHL,                  ///< Vector shift left
      VEC_SRL,                  ///< Vector shift right (logical)
      VEC_SRA,                  ///< Vector shift right (arithmetic)
      VEC_ROTL,                 ///< Vector rotate left
      VEC_ROTR,                 ///< Vector rotate right
      ROTQUAD_RZ_BYTES,         ///< Rotate quad right, by bytes, zero fill
      ROTQUAD_RZ_BITS,          ///< Rotate quad right, by bits, zero fill
      ROTBYTES_RIGHT_S,         ///< Vector rotate right, by bytes, sign fill
      ROTBYTES_LEFT,            ///< Rotate bytes (loads -> ROTQBYI)
      ROTBYTES_LEFT_CHAINED,    ///< Rotate bytes (loads -> ROTQBYI), with chain
      ROTBYTES_LEFT_BITS,       ///< Rotate bytes left by bit shift count
      SELECT_MASK,              ///< Select Mask (FSM, FSMB, FSMH, FSMBI)
      SELB,                     ///< Select bits -> (b & mask) | (a & ~mask)
      ADD_EXTENDED,             ///< Add extended, with carry
      CARRY_GENERATE,           ///< Carry generate for ADD_EXTENDED
      SUB_EXTENDED,             ///< Subtract extended, with borrow
      BORROW_GENERATE,          ///< Borrow generate for SUB_EXTENDED
      FPInterp,                 ///< Floating point interpolate
      FPRecipEst,               ///< Floating point reciprocal estimate
      SEXT32TO64,               ///< Sign-extended 32-bit const -> 64-bits
      LAST_SPUISD               ///< Last user-defined instruction
    };
  }

  /// Predicates that are used for node matching:
  namespace SPU {
    SDValue get_vec_u18imm(SDNode *N, SelectionDAG &DAG,
                             MVT ValueType);
    SDValue get_vec_i16imm(SDNode *N, SelectionDAG &DAG,
                             MVT ValueType);
    SDValue get_vec_i10imm(SDNode *N, SelectionDAG &DAG,
                             MVT ValueType);
    SDValue get_vec_i8imm(SDNode *N, SelectionDAG &DAG,
                            MVT ValueType);
    SDValue get_ILHUvec_imm(SDNode *N, SelectionDAG &DAG,
                              MVT ValueType);
    SDValue get_v4i32_imm(SDNode *N, SelectionDAG &DAG);
    SDValue get_v2i64_imm(SDNode *N, SelectionDAG &DAG);
  }

  class SPUTargetMachine;            // forward dec'l.
  
  class SPUTargetLowering :
    public TargetLowering
  {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int ReturnAddrIndex;              // FrameIndex for return slot.
    SPUTargetMachine &SPUTM;

  public:
    SPUTargetLowering(SPUTargetMachine &TM);
    
    /// getTargetNodeName() - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - Return the ValueType for ISD::SETCC
    virtual MVT getSetCCResultType(const SDValue &) const;
    
    //! Custom lowering hooks
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG);

    //! Provide custom lowering hooks for nodes with illegal result types.
    SDNode *ReplaceNodeResults(SDNode *N, SelectionDAG &DAG);
    
    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                                const APInt &Mask,
                                                APInt &KnownZero, 
                                                APInt &KnownOne,
                                                const SelectionDAG &DAG,
                                                unsigned Depth = 0) const;

    ConstraintType getConstraintType(const std::string &ConstraintLetter) const;

    std::pair<unsigned, const TargetRegisterClass*> 
      getRegForInlineAsmConstraint(const std::string &Constraint,
                                   MVT VT) const;

    void LowerAsmOperandForConstraint(SDValue Op, char ConstraintLetter,
                                      bool hasMemory, 
                                      std::vector<SDValue> &Ops,
                                      SelectionDAG &DAG) const;

    /// isLegalAddressImmediate - Return true if the integer value can be used
    /// as the offset of the target addressing mode.
    virtual bool isLegalAddressImmediate(int64_t V, const Type *Ty) const;
    virtual bool isLegalAddressImmediate(GlobalValue *) const;

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;
  };
}

#endif
