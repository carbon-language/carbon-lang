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
class PHINode : public Instruction {
  PHINode(const PHINode &PN);
public:
  PHINode(const Type *Ty, const string &Name = "");
  inline ~PHINode() { dropAllReferences(); }

  virtual Instruction *clone() const { return new PHINode(*this); }
  virtual string getOpcode() const { return "phi"; }

  // getNumIncomingValues - Return the number of incoming edges the PHI node has
  inline unsigned getNumIncomingValues() const { return Operands.size()/2; }

  // getIncomingValue - Return incoming value #x
  inline const Value *getIncomingValue(unsigned i) const {
    return Operands[i*2];
  }
  inline Value *getIncomingValue(unsigned i) {
    return Operands[i*2];
  }

  // getIncomingBlock - Return incoming basic block #x
  inline const BasicBlock *getIncomingBlock(unsigned i) const { 
    return Operands[i*2+1]->castBasicBlockAsserting();
  }
  inline BasicBlock *getIncomingBlock(unsigned i) { 
    return Operands[i*2+1]->castBasicBlockAsserting();
  }

  // addIncoming - Add an incoming value to the end of the PHI list
  void addIncoming(Value *D, BasicBlock *BB);

  // removeIncomingValue - Remove an incoming value.  This is useful if a
  // predecessor basic block is deleted.  The value removed is returned.
  Value *removeIncomingValue(const BasicBlock *BB);
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
  CallInst(const CallInst &CI);
public:
  CallInst(Method *M, vector<Value*> &params, const string &Name = "");
  inline ~CallInst() { dropAllReferences(); }

  virtual string getOpcode() const { return "call"; }

  virtual Instruction *clone() const { return new CallInst(*this); }
  bool hasSideEffects() const { return true; }


  const Method *getCalledMethod() const {
    return Operands[0]->castMethodAsserting(); 
  }
  Method *getCalledMethod() {
    return Operands[0]->castMethodAsserting(); 
  }
};

#endif
