//===-- llvm/BasicBlock.h - Represent a basic block in the VM ----*- C++ -*--=//
//
// This file contains the declaration of the BasicBlock class, which represents
// a single basic block in the VM.
//
// Note that basic blocks themselves are Def's, because they are referenced
// by instructions like branches and can go in switch tables and stuff...
//
// This may see wierd at first, but it's really pretty cool.  :)
//
//===----------------------------------------------------------------------===//
//
// Note that well formed basic blocks are formed of a list of instructions 
// followed by a single TerminatorInst instruction.  TerminatorInst's may not
// occur in the middle of basic blocks, and must terminate the blocks.
//
// This code allows malformed basic blocks to occur, because it may be useful
// in the intermediate stage of analysis or modification of a program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BASICBLOCK_H
#define LLVM_BASICBLOCK_H

#include "llvm/Value.h"               // Get the definition of Value
#include "llvm/ValueHolder.h"
#include "llvm/CFGdecls.h"

class Instruction;
class Method;
class BasicBlock;
class TerminatorInst;

typedef UseTy<BasicBlock> BasicBlockUse;

class BasicBlock : public Value {       // Basic blocks are data objects also
public:
  typedef ValueHolder<Instruction, BasicBlock> InstListType;
private :
  InstListType InstList;

  friend class ValueHolder<BasicBlock,Method>;
  void setParent(Method *parent);

public:
  typedef cfg::succ_iterator succ_iterator;   // Include CFG.h to use these
  typedef cfg::pred_iterator pred_iterator;
  typedef cfg::succ_const_iterator succ_const_iterator;
  typedef cfg::pred_const_iterator pred_const_iterator;

  BasicBlock(const string &Name = "", Method *Parent = 0);
  ~BasicBlock();

  // Specialize setName to take care of symbol table majik
  virtual void setName(const string &name);

  const Method *getParent() const { return (const Method*)InstList.getParent();}
        Method *getParent()       { return (Method*)InstList.getParent(); }

  const InstListType &getInstList() const { return InstList; }
        InstListType &getInstList()       { return InstList; }

  // getTerminator() - If this is a well formed basic block, then this returns
  // a pointer to the terminator instruction.  If it is not, then you get a null
  // pointer back.
  //
  TerminatorInst *getTerminator();
  const TerminatorInst *const getTerminator() const;

  // hasConstantPoolReferences() - This predicate is true if there is a 
  // reference to this basic block in the constant pool for this method.  For
  // example, if a block is reached through a switch table, that table resides
  // in the constant pool, and the basic block is reference from it.
  //
  bool hasConstantPoolReferences() const;

  // dropAllReferences() - This function causes all the subinstructions to "let
  // go" of all references that they are maintaining.  This allows one to
  // 'delete' a whole class at a time, even though there may be circular
  // references... first all references are dropped, and all use counts go to
  // zero.  Then everything is delete'd for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator 
  // delete.
  //
  void dropAllReferences();

  // splitBasicBlock - This splits a basic block into two at the specified
  // instruction.  Note that all instructions BEFORE the specified iterator stay
  // as part of the original basic block, an unconditional branch is added to 
  // the new BB, and the rest of the instructions in the BB are moved to the new
  // BB, including the old terminator.  The newly formed BasicBlock is returned.
  // This function invalidates the specified iterator.
  //
  // Note that this only works on well formed basic blocks (must have a 
  // terminator), and 'I' must not be the end of instruction list (which would
  // cause a degenerate basic block to be formed, having a terminator inside of
  // the basic block).
  //
  BasicBlock *splitBasicBlock(InstListType::iterator I);
};

#endif
