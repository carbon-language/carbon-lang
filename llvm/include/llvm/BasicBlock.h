//===-- llvm/BasicBlock.h - Represent a basic block in the VM ----*- C++ -*--=//
//
// This file contains the declaration of the BasicBlock class, which represents
// a single basic block in the VM.
//
// Note that basic blocks themselves are Value's, because they are referenced
// by instructions like branches and can go in switch tables and stuff...
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

#include "llvm/ValueHolder.h"
#include "llvm/Value.h"

class Instruction;
class Method;
class TerminatorInst;
class MachineCodeForBasicBlock;
template <class _Term, class _BB> class SuccIterator;  // Successor Iterator
template <class _Ptr, class _USE_iterator> class PredIterator;

class BasicBlock : public Value {       // Basic blocks are data objects also
public:
  typedef ValueHolder<Instruction, BasicBlock, Method> InstListType;
private :
  InstListType InstList;
  MachineCodeForBasicBlock* machineInstrVec;

  friend class ValueHolder<BasicBlock,Method,Method>;
  void setParent(Method *parent);

public:
  // Instruction iterators...
  typedef InstListType::iterator iterator;
  typedef InstListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  // Ctor, dtor
  BasicBlock(const std::string &Name = "", Method *Parent = 0);
  ~BasicBlock();

  // Specialize setName to take care of symbol table majik
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  // getParent - Return the enclosing method, or null if none
  const Method *getParent() const { return InstList.getParent(); }
        Method *getParent()       { return InstList.getParent(); }

  // getTerminator() - If this is a well formed basic block, then this returns
  // a pointer to the terminator instruction.  If it is not, then you get a null
  // pointer back.
  //
  TerminatorInst *getTerminator();
  const TerminatorInst *const getTerminator() const;
  
  // Machine code accessor...
  inline MachineCodeForBasicBlock& getMachineInstrVec() const {
    return *machineInstrVec; 
  }

  // Provide a scoped predecessor and successor iterator
  typedef PredIterator<BasicBlock, Value::use_iterator> pred_iterator;
  typedef PredIterator<const BasicBlock, 
                       Value::use_const_iterator> pred_const_iterator;

  typedef SuccIterator<TerminatorInst*, BasicBlock> succ_iterator;
  typedef SuccIterator<const TerminatorInst*,
                       const BasicBlock> succ_const_iterator;
  
  
  //===--------------------------------------------------------------------===//
  // Instruction iterator methods
  //
  inline iterator                begin()       { return InstList.begin(); }
  inline const_iterator          begin() const { return InstList.begin(); }
  inline iterator                end  ()       { return InstList.end();   }
  inline const_iterator          end  () const { return InstList.end();   }

  inline reverse_iterator       rbegin()       { return InstList.rbegin(); }
  inline const_reverse_iterator rbegin() const { return InstList.rbegin(); }
  inline reverse_iterator       rend  ()       { return InstList.rend();   }
  inline const_reverse_iterator rend  () const { return InstList.rend();   }

  inline unsigned                 size() const { return InstList.size(); }
  inline bool                    empty() const { return InstList.empty(); }
  inline const Instruction      *front() const { return InstList.front(); }
  inline       Instruction      *front()       { return InstList.front(); }
  inline const Instruction       *back()  const { return InstList.back(); }
  inline       Instruction       *back()        { return InstList.back(); }

  // getInstList() - Return the underlying instruction list container.  You need
  // to access it directly if you want to modify it currently.
  //
  const InstListType &getInstList() const { return InstList; }
        InstListType &getInstList()       { return InstList; }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BasicBlock *BB) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::BasicBlockVal;
  }

  // hasConstantReferences() - This predicate is true if there is a 
  // reference to this basic block in the constant pool for this method.  For
  // example, if a block is reached through a switch table, that table resides
  // in the constant pool, and the basic block is reference from it.
  //
  bool hasConstantReferences() const;

  // dropAllReferences() - This function causes all the subinstructions to "let
  // go" of all references that they are maintaining.  This allows one to
  // 'delete' a whole class at a time, even though there may be circular
  // references... first all references are dropped, and all use counts go to
  // zero.  Then everything is delete'd for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator 
  // delete.
  //
  void dropAllReferences();

  // removePredecessor - This method is used to notify a BasicBlock that the
  // specified Predecessor of the block is no longer able to reach it.  This is
  // actually not used to update the Predecessor list, but is actually used to 
  // update the PHI nodes that reside in the block.  Note that this should be
  // called while the predecessor still refers to this block.
  //
  void removePredecessor(BasicBlock *Pred);

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
  BasicBlock *splitBasicBlock(iterator I);
};

#endif
