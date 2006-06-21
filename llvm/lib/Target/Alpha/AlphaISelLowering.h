//===-- AlphaISelLowering.h - Alpha DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Andrew Lenharth and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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
      ITOFT_, FTOIT_, CVTQT_, CVTQS_, CVTTQ_,

      /// GPRelHi/GPRelLo - These represent the high and low 16-bit
      /// parts of a global address respectively.
      GPRelHi, GPRelLo, 

      /// RetLit - Literal Relocation of a Global
      RelLit,

      /// GlobalBaseReg - used to restore the GOT ptr
      GlobalBaseReg,

      /// GlobalRetAddr - used to restore the return address
      GlobalRetAddr,
      
      /// CALL - Normal call.
      CALL,

      /// DIVCALL - used for special library calls for div and rem
      DivCall,
      
      /// return flag operand
      RET_FLAG

    };
  }

  class AlphaTargetLowering : public TargetLowering {
    int VarArgsOffset;  // What is the offset to the first vaarg
    int VarArgsBase;    // What is the base FrameIndex
    unsigned GP; //GOT vreg
    unsigned RA; //Return Address
    bool useITOF;
  public:
    AlphaTargetLowering(TargetMachine &TM);
    
    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    virtual SDOperand CustomPromoteOperation(SDOperand Op, SelectionDAG &DAG);

    //Friendly names for dumps
    const char *getTargetNodeName(unsigned Opcode) const;

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg, unsigned CC,
                bool isTailCall, SDOperand Callee, ArgListTy &Args,
                SelectionDAG &DAG);

    ConstraintType getConstraintType(char ConstraintLetter) const;

    std::vector<unsigned> 
      getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                        MVT::ValueType VT) const;

    void restoreGP(MachineBasicBlock* BB);
    void restoreRA(MachineBasicBlock* BB);
    unsigned getVRegGP() { return GP; }
    unsigned getVRegRA() { return RA; }
    bool hasITOF() { return useITOF; }
  };
}

#endif   // LLVM_TARGET_ALPHA_ALPHAISELLOWERING_H
