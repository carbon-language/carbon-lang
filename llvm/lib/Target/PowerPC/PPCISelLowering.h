//===-- PPC32ISelLowering.h - PPC32 DAG Lowering Interface ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PPC uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_POWERPC_PPC32ISELLOWERING_H
#define LLVM_TARGET_POWERPC_PPC32ISELLOWERING_H

#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "PowerPC.h"

namespace llvm {
  namespace PPCISD {
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+PPC::INSTRUCTION_LIST_END,

      /// FSEL - Traditional three-operand fsel node.
      ///
      FSEL,
    };
  }  
  
  class PPC32TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    int ReturnAddrIndex;              // FrameIndex for return slot.
  public:
    PPC32TargetLowering(TargetMachine &TM);
    
    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    
    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
      LowerArguments(Function &F, SelectionDAG &DAG);
    
    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
      LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
                  unsigned CC,
                  bool isTailCall, SDOperand Callee, ArgListTy &Args,
                  SelectionDAG &DAG);
    
    virtual SDOperand LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                   Value *VAListV, SelectionDAG &DAG);
    
    virtual std::pair<SDOperand,SDOperand>
      LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
                 const Type *ArgTy, SelectionDAG &DAG);
    
    virtual std::pair<SDOperand, SDOperand>
      LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                              SelectionDAG &DAG);
    
    virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB);
  };
}

#endif   // LLVM_TARGET_POWERPC_PPC32ISELLOWERING_H
