//===-- BasicBlock.cpp - Implement BasicBlock related functions --*- C++ -*--=//
//
// This file implements the Method class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/ValueHolderImpl.h"
#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/SymbolTable.h"
#include "llvm/Type.h"

// Instantiate Templates - This ugliness is the price we have to pay
// for having a ValueHolderImpl.h file seperate from ValueHolder.h!  :(
//
template class ValueHolder<Instruction, BasicBlock>;

BasicBlock::BasicBlock(const string &name, Method *parent)
  : Value(Type::LabelTy, Value::BasicBlockVal, name), InstList(this, 0) {

  if (parent)
    parent->getBasicBlocks().push_back(this);
}

BasicBlock::~BasicBlock() {
  dropAllReferences();
  InstList.delete_all();
}

// Specialize setName to take care of symbol table majik
void BasicBlock::setName(const string &name) {
  Method *P;
  if ((P = getParent()) && hasName()) P->getSymbolTable()->remove(this);
  Value::setName(name);
  if (P && hasName()) P->getSymbolTable()->insert(this);
}

void BasicBlock::setParent(Method *parent) { 
  if (getParent() && hasName())
    getParent()->getSymbolTable()->remove(this);

  InstList.setParent(parent);

  if (getParent() && hasName())
    getParent()->getSymbolTableSure()->insert(this);
}

TerminatorInst *BasicBlock::getTerminator() {
  if (InstList.empty()) return 0;
  Instruction *T = InstList.back();
  if (T->isTerminator()) return (TerminatorInst*)T;
  return 0;
}

const TerminatorInst *const BasicBlock::getTerminator() const {
  if (InstList.empty()) return 0;
  const Instruction *T = InstList.back();
  if (T->isTerminator()) return (TerminatorInst*)T;
  return 0;
}

void BasicBlock::dropAllReferences() {
  for_each(InstList.begin(), InstList.end(), 
	   std::mem_fun(&Instruction::dropAllReferences));
}

// hasConstantPoolReferences() - This predicate is true if there is a 
// reference to this basic block in the constant pool for this method.  For
// example, if a block is reached through a switch table, that table resides
// in the constant pool, and the basic block is reference from it.
//
bool BasicBlock::hasConstantPoolReferences() const {
  for (use_const_iterator I = use_begin(), E = use_end(); I != E; ++I)
    if ((*I)->getValueType() == ConstantVal)
      return true;

  return false;
}


// splitBasicBlock - This splits a basic block into two at the specified
// instruction.  Note that all instructions BEFORE the specified iterator stay
// as part of the original basic block, an unconditional branch is added to 
// the new BB, and the rest of the instructions in the BB are moved to the new
// BB, including the old terminator.  This invalidates the iterator.
//
// Note that this only works on well formed basic blocks (must have a 
// terminator), and 'I' must not be the end of instruction list (which would
// cause a degenerate basic block to be formed, having a terminator inside of
// the basic block). 
//
BasicBlock *BasicBlock::splitBasicBlock(InstListType::iterator I) {
  assert(getTerminator() && "Can't use splitBasicBlock on degenerate BB!");
  assert(I != InstList.end() && 
	 "Trying to get me to create degenerate basic block!");

  BasicBlock *New = new BasicBlock("", getParent());

  // Go from the end of the basic block through to the iterator pointer, moving
  // to the new basic block...
  Instruction *Inst = 0;
  do {
    InstListType::iterator EndIt = InstList.end();
    Inst = InstList.remove(--EndIt);                  // Remove from end
    New->InstList.push_front(Inst);                   // Add to front
  } while (Inst != *I);   // Loop until we move the specified instruction.

  // Add a branch instruction to the newly formed basic block.
  InstList.push_back(new BranchInst(New));
  return New;
}
