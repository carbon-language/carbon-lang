//===-- BasicBlock.cpp - Implement BasicBlock related methods -------------===//
//
// This file implements the BasicBlock class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/BasicBlock.h"
#include "llvm/iTerminators.h"
#include "llvm/Type.h"
#include "llvm/Support/CFG.h"
#include "llvm/Constant.h"
#include "llvm/iPHINode.h"
#include "llvm/SymbolTable.h"
#include "Support/LeakDetector.h"
#include "SymbolTableListTraitsImpl.h"
#include <algorithm>

// DummyInst - An instance of this class is used to mark the end of the
// instruction list.  This is not a real instruction.
//
struct DummyInst : public Instruction {
  DummyInst() : Instruction(Type::VoidTy, OtherOpsEnd) {
    // This should not be garbage monitored.
    LeakDetector::removeGarbageObject(this);
  }

  virtual Instruction *clone() const {
    assert(0 && "Cannot clone EOL");abort();
    return 0;
  }
  virtual const char *getOpcodeName() const { return "*end-of-list-inst*"; }

  // Methods for support type inquiry through isa, cast, and dyn_cast...
  static inline bool classof(const DummyInst *) { return true; }
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == OtherOpsEnd;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

Instruction *ilist_traits<Instruction>::createNode() {
  return new DummyInst();
}
iplist<Instruction> &ilist_traits<Instruction>::getList(BasicBlock *BB) {
  return BB->getInstList();
}

// Explicit instantiation of SymbolTableListTraits since some of the methods
// are not in the public header file...
template SymbolTableListTraits<Instruction, BasicBlock, Function>;


// BasicBlock ctor - If the function parameter is specified, the basic block is
// automatically inserted at the end of the function.
//
BasicBlock::BasicBlock(const std::string &name, Function *Parent)
  : Value(Type::LabelTy, Value::BasicBlockVal, name) {
  // Initialize the instlist...
  InstList.setItemParent(this);

  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (Parent)
    Parent->getBasicBlockList().push_back(this);
}

/// BasicBlock ctor - If the InsertBefore parameter is specified, the basic
/// block is automatically inserted right before the specified block.
///
BasicBlock::BasicBlock(const std::string &Name, BasicBlock *InsertBefore)
  : Value(Type::LabelTy, Value::BasicBlockVal, Name) {
  // Initialize the instlist...
  InstList.setItemParent(this);

  // Make sure that we get added to a function
  LeakDetector::addGarbageObject(this);

  if (InsertBefore) {
    assert(InsertBefore->getParent() &&
           "Cannot insert block before another block that is not embedded into"
           " a function yet!");
    InsertBefore->getParent()->getBasicBlockList().insert(InsertBefore, this);
  }
}


BasicBlock::~BasicBlock() {
  dropAllReferences();
  InstList.clear();
}

void BasicBlock::setParent(Function *parent) {
  if (getParent())
    LeakDetector::addGarbageObject(this);

  InstList.setParent(parent);

  if (getParent())
    LeakDetector::removeGarbageObject(this);
}

// Specialize setName to take care of symbol table majik
void BasicBlock::setName(const std::string &name, SymbolTable *ST) {
  Function *P;
  assert((ST == 0 || (!getParent() || ST == &getParent()->getSymbolTable())) &&
	 "Invalid symtab argument!");
  if ((P = getParent()) && hasName()) P->getSymbolTable().remove(this);
  Value::setName(name);
  if (P && hasName()) P->getSymbolTable().insert(this);
}

TerminatorInst *BasicBlock::getTerminator() {
  if (InstList.empty()) return 0;
  return dyn_cast<TerminatorInst>(&InstList.back());
}

const TerminatorInst *const BasicBlock::getTerminator() const {
  if (InstList.empty()) return 0;
  return dyn_cast<TerminatorInst>(&InstList.back());
}

void BasicBlock::dropAllReferences() {
  for(iterator I = begin(), E = end(); I != E; ++I)
    I->dropAllReferences();
}

// hasConstantReferences() - This predicate is true if there is a 
// reference to this basic block in the constant pool for this method.  For
// example, if a block is reached through a switch table, that table resides
// in the constant pool, and the basic block is reference from it.
//
bool BasicBlock::hasConstantReferences() const {
  for (use_const_iterator I = use_begin(), E = use_end(); I != E; ++I)
    if (::isa<Constant>((Value*)*I))
      return true;

  return false;
}

// removePredecessor - This method is used to notify a BasicBlock that the
// specified Predecessor of the block is no longer able to reach it.  This is
// actually not used to update the Predecessor list, but is actually used to 
// update the PHI nodes that reside in the block.  Note that this should be
// called while the predecessor still refers to this block.
//
void BasicBlock::removePredecessor(BasicBlock *Pred) {
  assert(find(pred_begin(this), pred_end(this), Pred) != pred_end(this) &&
	 "removePredecessor: BB is not a predecessor!");
  if (!isa<PHINode>(front())) return;   // Quick exit.

  pred_iterator PI(pred_begin(this)), EI(pred_end(this));
  unsigned max_idx;

  // Loop over the rest of the predecessors until we run out, or until we find
  // out that there are more than 2 predecessors.
  for (max_idx = 0; PI != EI && max_idx < 3; ++PI, ++max_idx) /*empty*/;

  // If there are exactly two predecessors, then we want to nuke the PHI nodes
  // altogether.  We cannot do this, however if this in this case however:
  //
  //  Loop:
  //    %x = phi [X, Loop]
  //    %x2 = add %x, 1         ;; This would become %x2 = add %x2, 1
  //    br Loop                 ;; %x2 does not dominate all uses
  //
  // This is because the PHI node input is actually taken from the predecessor
  // basic block.  The only case this can happen is with a self loop, so we 
  // check for this case explicitly now.
  // 
  assert(max_idx != 0 && "PHI Node in block with 0 predecessors!?!?!");
  if (max_idx == 2) {
    PI = pred_begin(this);
    BasicBlock *Other = *PI == Pred ? *++PI : *PI;

    // Disable PHI elimination!
    if (this == Other) max_idx = 3;
  }

  if (max_idx <= 2) {                // <= Two predecessors BEFORE I remove one?
    // Yup, loop through and nuke the PHI nodes
    while (PHINode *PN = dyn_cast<PHINode>(&front())) {
      PN->removeIncomingValue(Pred); // Remove the predecessor first...

      // If the PHI _HAD_ two uses, replace PHI node with its now *single* value
      if (max_idx == 2) {
        if (PN->getOperand(0) != PN)
          PN->replaceAllUsesWith(PN->getOperand(0));
        else
          // We are left with an infinite loop with no entries: kill the PHI.
          PN->replaceAllUsesWith(Constant::getNullValue(PN->getType()));
        getInstList().pop_front();    // Remove the PHI node
      }

      // If the PHI node already only had one entry, it got deleted by
      // removeIncomingValue.
    }
  } else {
    // Okay, now we know that we need to remove predecessor #pred_idx from all
    // PHI nodes.  Iterate over each PHI node fixing them up
    for (iterator II = begin(); PHINode *PN = dyn_cast<PHINode>(II); ++II)
      PN->removeIncomingValue(Pred);
  }
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
BasicBlock *BasicBlock::splitBasicBlock(iterator I, const std::string &BBName) {
  assert(getTerminator() && "Can't use splitBasicBlock on degenerate BB!");
  assert(I != InstList.end() && 
	 "Trying to get me to create degenerate basic block!");

  BasicBlock *New = new BasicBlock(BBName, getParent());

  // Go from the end of the basic block through to the iterator pointer, moving
  // to the new basic block...
  Instruction *Inst = 0;
  do {
    iterator EndIt = end();
    Inst = InstList.remove(--EndIt);                  // Remove from end
    New->InstList.push_front(Inst);                   // Add to front
  } while (Inst != &*I);   // Loop until we move the specified instruction.

  // Add a branch instruction to the newly formed basic block.
  InstList.push_back(new BranchInst(New));

  // Now we must loop through all of the successors of the New block (which
  // _were_ the successors of the 'this' block), and update any PHI nodes in
  // successors.  If there were PHI nodes in the successors, then they need to
  // know that incoming branches will be from New, not from Old.
  //
  for (BasicBlock::succ_iterator I = succ_begin(New), E = succ_end(New);
       I != E; ++I) {
    // Loop over any phi nodes in the basic block, updating the BB field of
    // incoming values...
    BasicBlock *Successor = *I;
    for (BasicBlock::iterator II = Successor->begin();
         PHINode *PN = dyn_cast<PHINode>(II); ++II) {
      int IDX = PN->getBasicBlockIndex(this);
      while (IDX != -1) {
        PN->setIncomingBlock((unsigned)IDX, New);
        IDX = PN->getBasicBlockIndex(this);
      }
    }
  }
  return New;
}
