//===-- llvm/Support/CallSite.h - Abstract Call & Invoke instrs -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
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

  /// CallSite::get - This static method is sort of like a constructor.  It will
  /// create an appropriate call site for a Call or Invoke instruction, but it
  /// can also create a null initialized CallSite object for something which is
  /// NOT a call site.
  ///
  static CallSite get(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (I->getOpcode() == Instruction::Call)
        return CallSite((CallInst*)I);
      else if (I->getOpcode() == Instruction::Invoke)
        return CallSite((InvokeInst*)I);
    }
    return CallSite();
  }

  /// getInstruction - Return the instruction this call site corresponds to
  ///
  Instruction *getInstruction() const { return I; }

  /// getCalledValue - Return the pointer to function that is being called...
  ///
  Value *getCalledValue() const {
    assert(I && "Not a call or invoke instruction!");
    return I->getOperand(0);
  }

  /// getCalledFunction - Return the function being called if this is a direct
  /// call, otherwise return null (if it's an indirect call).
  ///
  /// FIXME: This should be inlined once ConstantPointerRefs are gone.  :(
  Function *getCalledFunction() const;

  /// setCalledFunction - Set the callee to the specified value...
  ///
  void setCalledFunction(Value *V) {
    assert(I && "Not a call or invoke instruction!");
    I->setOperand(0, V);
  }

  /// arg_iterator - The type of iterator to use when looping over actual
  /// arguments at this call site...
  typedef User::op_iterator arg_iterator;

  /// arg_begin/arg_end - Return iterators corresponding to the actual argument
  /// list for a call site.
  ///
  arg_iterator arg_begin() const {
    assert(I && "Not a call or invoke instruction!");
    if (I->getOpcode() == Instruction::Call)
      return I->op_begin()+1; // Skip Function
    else
      return I->op_begin()+3; // Skip Function, BB, BB
  }
  arg_iterator arg_end() const { return I->op_end(); }
};

#endif
