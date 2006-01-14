//===-- IA64ISelLowering.h - IA64 DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that IA64 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_IA64_IA64ISELLOWERING_H
#define LLVM_TARGET_IA64_IA64ISELLOWERING_H

#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "IA64.h"

namespace llvm {
  namespace IA64ISD {
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+IA64::INSTRUCTION_LIST_END,

      /// GETFD - the getf.d instruction takes a floating point operand and
      /// returns its 64-bit memory representation as an i64
      GETFD,

      // TODO: explain this hack
      BRCALL
    };
  }  
  
  class IA64TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    //int ReturnAddrIndex;              // FrameIndex for return slot.
    unsigned GP, SP, RP; // FIXME - clean this mess up
	  
  public:
    IA64TargetLowering(TargetMachine &TM);

    unsigned VirtGPR; // this is public so it can be accessed in the selector
                      // for ISD::RET. add an accessor instead? FIXME
	    
    /// LowerOperation - Provide custom lowering hooks for some operations.
    ///
// XXX    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    
    const char *getTargetNodeName(unsigned Opcode) const;
      
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
    
// XXX    virtual MachineBasicBlock *InsertAtEndOfBasicBlock(MachineInstr *MI,
// XXX                                                      MachineBasicBlock *MBB);
  };
}

#endif   // LLVM_TARGET_IA64_IA64ISELLOWERING_H
