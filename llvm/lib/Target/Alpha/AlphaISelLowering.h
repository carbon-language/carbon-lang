//===-- AlphaISelLowering.h - Alpha DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Alpha uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ALPHA_ALPHAISELLOWERING_H
#define LLVM_TARGET_ALPHA_ALPHAISELLOWERING_H

#include "llvm/ADT/VectorExtras.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "Alpha.h"

namespace llvm {

  namespace AlphaISD {
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+Alpha::INSTRUCTION_LIST_END,
      //These corrospond to the identical Instruction
      CVTQT_, CVTQS_, CVTTQ_,

      /// GPRelHi/GPRelLo - These represent the high and low 16-bit
      /// parts of a global address respectively.
      GPRelHi, GPRelLo, 

      /// RetLit - Literal Relocation of a Global
      RelLit,

      /// GlobalRetAddr - used to restore the return address
      GlobalRetAddr,
      
      /// CALL - Normal call.
      CALL,

      /// DIVCALL - used for special library calls for div and rem
      DivCall,
      
      /// return flag operand
      RET_FLAG,

      /// CHAIN = COND_BRANCH CHAIN, OPC, (G|F)PRC, DESTBB [, INFLAG] - This
      /// corresponds to the COND_BRANCH pseudo instruction.  
      /// *PRC is the input register to compare to zero,
      /// OPC is the branch opcode to use (e.g. Alpha::BEQ),
      /// DESTBB is the destination block to branch to, and INFLAG is
      /// an optional input flag argument.
      COND_BRANCH_I, COND_BRANCH_F

    };
  }

  class AlphaTargetLowering : public TargetLowering {
    int VarArgsOffset;  // What is the offset to the first vaarg
    int VarArgsBase;    // What is the base FrameIndex
    bool useITOF;
  public:
    explicit AlphaTargetLowering(TargetMachine &TM);
    
    /// getSetCCResultType - Get the SETCC result ValueType
    virtual MVT getSetCCResultType(const SDOperand &) const;

    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    virtual SDOperand CustomPromoteOperation(SDOperand Op, SelectionDAG &DAG);

    //Friendly names for dumps
    const char *getTargetNodeName(unsigned Opcode) const;

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool RetSExt, bool RetZExt,
                bool isVarArg, unsigned CC, bool isTailCall, SDOperand Callee, 
                ArgListTy &Args, SelectionDAG &DAG);

    ConstraintType getConstraintType(const std::string &Constraint) const;

    std::vector<unsigned> 
      getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                        MVT VT) const;

    bool hasITOF() { return useITOF; }

    MachineBasicBlock *EmitInstrWithCustomInserter(MachineInstr *MI,
                                                   MachineBasicBlock *BB);
  };
}

#endif   // LLVM_TARGET_ALPHA_ALPHAISELLOWERING_H
