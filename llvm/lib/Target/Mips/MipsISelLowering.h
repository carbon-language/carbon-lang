//===-- MipsISelLowering.h - Mips DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Mips uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef MipsISELLOWERING_H
#define MipsISELLOWERING_H

#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
#include "Mips.h"
#include "MipsSubtarget.h"

namespace llvm {
  namespace MipsISD {
    enum NodeType {
      // Start the numbering from where ISD NodeType finishes.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+Mips::INSTRUCTION_LIST_END,

      // Jump and link (call)
      JmpLink,

      // Get the Higher 16 bits from a 32-bit immediate
      // No relation with Mips Hi register
      Hi, 

      // Get the Lower 16 bits from a 32-bit immediate
      // No relation with Mips Lo register
      Lo, 

      // Return 
      Ret
    };
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Implementation
  //===--------------------------------------------------------------------===//
  class MipsTargetLowering : public TargetLowering 
  {
    // FrameIndex for return slot.
    int ReturnAddrIndex;

    // const MipsSubtarget &MipsSubTarget;
  public:

    MipsTargetLowering(MipsTargetMachine &TM);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

    /// getTargetNodeName - This method returns the name of a target specific 
    //  DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

  private:
    // Lower Operand helpers
    SDOperand LowerCCCArguments(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCCCCallTo(SDOperand Op, SelectionDAG &DAG, unsigned CC);
    SDNode *LowerCallResult(SDOperand Chain, SDOperand InFlag, SDNode*TheCall,
                            unsigned CallingConv, SelectionDAG &DAG);
    SDOperand getReturnAddressFrameIndex(SelectionDAG &DAG);

    // Lower Operand specifics
    SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerGlobalTLSAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerRETURNADDR(SDOperand Op, SelectionDAG &DAG);

  };
}

#endif // MipsISELLOWERING_H
