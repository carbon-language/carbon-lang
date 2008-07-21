//===-- MipsISelLowering.h - Mips DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

      // Handle gp_rel (small data/bss sections) relocation.
      GPRel,

      // Select CC Pseudo Instruction
      SelectCC,

      // Floating Point Branch Conditional
      FPBrcond,

      // Floating Point Compare
      FPCmp,

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

    explicit MipsTargetLowering(MipsTargetMachine &TM);

    /// LowerOperation - Provide custom lowering hooks for some operations.
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);

    /// getTargetNodeName - This method returns the name of a target specific 
    //  DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - get the ISD::SETCC result ValueType
    MVT getSetCCResultType(const SDOperand &) const;

  private:
    // Subtarget Info
    const MipsSubtarget *Subtarget;

    // Lower Operand helpers
    SDOperand LowerCCCArguments(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCCCCallTo(SDOperand Op, SelectionDAG &DAG, unsigned CC);
    SDNode *LowerCallResult(SDOperand Chain, SDOperand InFlag, SDNode*TheCall,
                            unsigned CallingConv, SelectionDAG &DAG);
    bool IsGlobalInSmallSection(GlobalValue *GV); 

    // Lower Operand specifics
    SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerGlobalAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerGlobalTLSAddress(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerJumpTable(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerConstantPool(SDOperand Op, SelectionDAG &DAG);
    SDOperand LowerSELECT_CC(SDOperand Op, SelectionDAG &DAG);

    virtual MachineBasicBlock *EmitInstrWithCustomInserter(MachineInstr *MI,
                                                        MachineBasicBlock *MBB);

    // Inline asm support
    ConstraintType getConstraintType(const std::string &Constraint) const;

    std::pair<unsigned, const TargetRegisterClass*> 
              getRegForInlineAsmConstraint(const std::string &Constraint,
              MVT VT) const;

    std::vector<unsigned>
    getRegClassForInlineAsmConstraint(const std::string &Constraint,
              MVT VT) const;
  };
}

#endif // MipsISELLOWERING_H
