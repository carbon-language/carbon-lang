//===-- InstrTypes.cpp - Implement Instruction subclasses --------*- C++ -*--=//
//
// This file implements 
//
//===----------------------------------------------------------------------===//

#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/SymbolTable.h"
#include "llvm/Type.h"
#include <algorithm>

// TODO: Move to getUnaryOperator iUnary.cpp when and if it exists!
UnaryOperator *UnaryOperator::create(unsigned Op, Value *Source) {
  switch (Op) {
  default:
    cerr << "Don't know how to GetUnaryOperator " << Op << endl;
    return 0;
  }
}

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

TerminatorInst::TerminatorInst(unsigned iType) 
  : Instruction(Type::VoidTy, iType, "") {
}


//===----------------------------------------------------------------------===//
//                            MethodArgument Class
//===----------------------------------------------------------------------===//

// Specialize setName to take care of symbol table majik
void MethodArgument::setName(const string &name) {
  Method *P;
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && hasName()) P->getSymbolTable()->insert(this);
}


//===----------------------------------------------------------------------===//
//                               PHINode Class
//===----------------------------------------------------------------------===//

PHINode::PHINode(const Type *Ty, const string &name) 
  : Instruction(Ty, Instruction::PHINode, name) {
}

PHINode::PHINode(const PHINode &PN) 
  : Instruction(PN.getType(), Instruction::PHINode) {
  
  for (unsigned i = 0; i < PN.IncomingValues.size(); i++)
    IncomingValues.push_back(
	make_pair(Use(PN.IncomingValues[i].first, this),
		  BasicBlockUse(PN.IncomingValues[i].second, this)));
}

void PHINode::dropAllReferences() {
  IncomingValues.clear();
}

bool PHINode::setOperand(unsigned i, Value *Val) {
  assert(Val && "PHI node must only reference nonnull definitions!");
  if (i >= IncomingValues.size()*2) return false;

  if (i & 1) {
    IncomingValues[i/2].second = Val->castBasicBlockAsserting();
  } else {
    IncomingValues[i/2].first  = Val;
  }
  return true;
}

void PHINode::addIncoming(Value *D, BasicBlock *BB) {
  IncomingValues.push_back(make_pair(Use(D, this), BasicBlockUse(BB, this)));
}

struct FindBBEntry {
  const BasicBlock *BB;
  inline FindBBEntry(const BasicBlock *bb) : BB(bb) {}
  inline bool operator()(const pair<Use,BasicBlockUse> &Entry) {
    return Entry.second == BB;
  }
};


// removeIncomingValue - Remove an incoming value.  This is useful if a
// predecessor basic block is deleted.
Value *PHINode::removeIncomingValue(const BasicBlock *BB) {
  vector<PairTy>::iterator Idx = find_if(IncomingValues.begin(), 
					 IncomingValues.end(), FindBBEntry(BB));
  assert(Idx != IncomingValues.end() && "BB not in PHI node!");
  Value *Removed = Idx->first;
  IncomingValues.erase(Idx);
  return Removed;
}
