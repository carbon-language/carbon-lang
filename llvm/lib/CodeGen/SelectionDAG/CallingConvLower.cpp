//===-- llvm/CallingConvLower.cpp - Calling Conventions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CCState class, used for lowering and implementing
// calling conventions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

CCState::CCState(unsigned CC, const TargetMachine &tm,
                 SmallVector<CCValAssign, 16> &locs)
  : CallingConv(CC), TM(tm), MRI(*TM.getRegisterInfo()), Locs(locs) {
  // No stack is used.
  StackOffset = 0;
  
  UsedRegs.resize(MRI.getNumRegs());
}


/// MarkAllocated - Mark a register and all of its aliases as allocated.
void CCState::MarkAllocated(unsigned Reg) {
  UsedRegs[Reg/32] |= 1 << (Reg&31);
  
  if (const unsigned *RegAliases = MRI.getAliasSet(Reg))
    for (; (Reg = *RegAliases); ++RegAliases)
      UsedRegs[Reg/32] |= 1 << (Reg&31);
}

/// AnalyzeCallOperands - Analyze an ISD::CALL node, incorporating info
/// about the passed values into this state.
void CCState::AnalyzeCallOperands(SDNode *TheCall, CCAssignFn Fn) {
  unsigned NumOps = (TheCall->getNumOperands() - 5) / 2;
  for (unsigned i = 0; i != NumOps; ++i) {
    MVT::ValueType ArgVT = TheCall->getOperand(5+2*i).getValueType();
    SDOperand FlagOp = TheCall->getOperand(5+2*i+1);
    unsigned ArgFlags =cast<ConstantSDNode>(FlagOp)->getValue();
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
      cerr << "Call operand #" << i << " has unhandled type "
           << MVT::getValueTypeString(ArgVT) << "\n";
      abort();
    }
  }
}

/// AnalyzeFormalArguments - Analyze an ISD::FORMAL_ARGUMENTS node,
/// incorporating info about the formals into this state.
void CCState::AnalyzeFormalArguments(SDNode *TheArgs, CCAssignFn Fn) {
  unsigned NumArgs = TheArgs->getNumValues()-1;

  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT::ValueType ArgVT = TheArgs->getValueType(i);
    SDOperand FlagOp = TheArgs->getOperand(3+i);
    unsigned ArgFlags = cast<ConstantSDNode>(FlagOp)->getValue();
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
      cerr << "Formal argument #" << i << " has unhandled type "
           << MVT::getValueTypeString(ArgVT) << "\n";
      abort();
    }
  }
}
