//===-- llvm/Support/CallSite.h - Abstract Call & Invoke instrs -*- C++ -*-===//
//
// This file defines the CallSite class, which is a handy wrapper for code that
// wants to treat Call and Invoke instructions in a generic way.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CALLSITE_H
#define LLVM_SUPPORT_CALLSITE_H

#include "llvm/Instruction.h"

class CallInst;
class InvokeInst;

class CallSite {
  Instruction *I;
public:
  CallSite() : I(0) {}
  CallSite(CallInst *CI) : I((Instruction*)CI) {}
  CallSite(InvokeInst *II) : I((Instruction*)II) {}
  CallSite(const CallSite &CS) : I(CS.I) {}
  CallSite &operator=(const CallSite &CS) { I = CS.I; return *this; }

  /// getInstruction - Return the instruction this call site corresponds to
  Instruction *getInstruction() const { return I; }

  /// getCalledValue - Return the pointer to function that is being called...
  ///
  Value *getCalledValue() const { return I->getOperand(0); }

  /// getCalledFunction - Return the function being called if this is a direct
  /// call, otherwise return null (if it's an indirect call).
  ///
  Function *getCalledFunction() const {
    return dyn_cast<Function>(getCalledValue());
  }

  /// arg_iterator - The type of iterator to use when looping over actual
  /// arguments at this call site...
  typedef User::op_iterator arg_iterator;

  /// arg_begin/arg_end - Return iterators corresponding to the actual argument
  /// list for a call site.
  ///
  arg_iterator arg_begin() const {
    if (I->getOpcode() == Instruction::Call)
      return I->op_begin()+1; // Skip Function
    else
      return I->op_begin()+3; // Skip Function, BB, BB
  }
  arg_iterator arg_end() const { return I->op_begin(); }
};

#endif
