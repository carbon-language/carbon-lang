//===-- llvm/iOther.h - "Other" instruction node definitions -----*- C++ -*--=//
//
// This file contains the declarations for instructions that fall into the 
// grandios 'other' catagory...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IOTHER_H
#define LLVM_IOTHER_H

#include "llvm/InstrTypes.h"
#include "llvm/Method.h"
#include <vector>

//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

// PHINode - The PHINode class is used to represent the magical mystical PHI
// node, that can not exist in nature, but can be synthesized in a computer
// scientist's overactive imagination.
//
// TODO: FIXME: This representation is not good enough.  Consider the following
//       code:
//       BB0: %x = int %0
//       BB1: %y = int %1
//       BB2: %z = phi int %0, %1 - Can't tell where constants come from!
//
// TOFIX: Store pair<Use,BasicBlockUse> instead of just <Use>
//
class PHINode : public Instruction {
  vector<Use> IncomingValues;
  PHINode(const PHINode &PN);
public:
  PHINode(const Type *Ty, const string &Name = "");
  inline ~PHINode() { dropAllReferences(); }

  virtual Instruction *clone() const { return new PHINode(*this); }

  // Implement all of the functionality required by User...
  //
  virtual void dropAllReferences();
  virtual const Value *getOperand(unsigned i) const { 
    return (i < IncomingValues.size()) ? IncomingValues[i] : 0; 
  }
  inline Value *getOperand(unsigned i) {
    return (Value*)((const PHINode*)this)->getOperand(i);
  }
  virtual unsigned getNumOperands() const { return IncomingValues.size(); }
  virtual bool setOperand(unsigned i, Value *Val);
  virtual string getOpcode() const { return "phi"; }

  void addIncoming(Value *D);
};


//===----------------------------------------------------------------------===//
//                           MethodArgument Class
//===----------------------------------------------------------------------===//

class MethodArgument : public Value {  // Defined in the InstrType.cpp file
  Method *Parent;

  friend class ValueHolder<MethodArgument,Method>;
  inline void setParent(Method *parent) { Parent = parent; }

public:
  MethodArgument(const Type *Ty, const string &Name = "") 
    : Value(Ty, Value::MethodArgumentVal, Name) {
    Parent = 0;
  }

  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name);

  inline const Method *getParent() const { return Parent; }
  inline       Method *getParent()       { return Parent; }
};


//===----------------------------------------------------------------------===//
//             Classes to function calls and method invocations
//===----------------------------------------------------------------------===//

class CallInst : public Instruction {
  MethodUse M;
  vector<Use> Params;
  CallInst(const CallInst &CI);
public:
  CallInst(Method *M, vector<Value*> &params, const string &Name = "");
  inline ~CallInst() { dropAllReferences(); }

  virtual string getOpcode() const { return "call"; }

  virtual Instruction *clone() const { return new CallInst(*this); }
  bool hasSideEffects() const { return true; }


  const Method *getCalledMethod() const { return M; }
        Method *getCalledMethod()       { return M; }

  // Implement all of the functionality required by Instruction...
  //
  virtual void dropAllReferences();
  virtual const Value *getOperand(unsigned i) const { 
    return i == 0 ? M : ((i <= Params.size()) ? Params[i-1] : 0);
  }
  inline Value *getOperand(unsigned i) {
    return (Value*)((const CallInst*)this)->getOperand(i);
  }
  virtual unsigned getNumOperands() const { return Params.size()+1; }

  virtual bool setOperand(unsigned i, Value *Val);
};

#endif
