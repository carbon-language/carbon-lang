//===-- IA64ISelLowering.h - IA64 DAG Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
      BRCALL,
      
      // RET_FLAG - Return with a flag operand
      RET_FLAG
    };
  }  
  
  class IA64TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    //int ReturnAddrIndex;              // FrameIndex for return slot.
    unsigned GP, SP, RP; // FIXME - clean this mess up
  public:
    explicit IA64TargetLowering(TargetMachine &TM);

    unsigned VirtGPR; // this is public so it can be accessed in the selector
                      // for ISD::RET. add an accessor instead? FIXME
    const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType: return ISD::SETCC's result type.
    virtual MVT getSetCCResultType(const SDOperand &) const;
      
    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
      LowerArguments(Function &F, SelectionDAG &DAG);
    
    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
      LowerCallTo(SDOperand Chain, const Type *RetTy,
                  bool RetSExt, bool RetZExt, bool isVarArg,
                  unsigned CC, bool isTailCall, 
                  SDOperand Callee, ArgListTy &Args, SelectionDAG &DAG);

    /// LowerOperation - for custom lowering specific ops
    /// (currently, only "ret void")
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    
  };
}

#endif   // LLVM_TARGET_IA64_IA64ISELLOWERING_H
