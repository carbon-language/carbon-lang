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
#include "llvm/InstrTypes.h"
#include <list>

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

  //===--------------------------------------------------------------------===//
  // Predecessor iterator code
  //===--------------------------------------------------------------------===//
  // 
  // This is used to figure out what basic blocks we could be coming from.
  //

  // Forward declare iterator class template...
  template <class _Ptr, class _USE_iterator> class PredIterator;

  typedef PredIterator<BasicBlock*, use_iterator> pred_iterator;
  typedef PredIterator<const BasicBlock*, 
                       use_const_iterator> pred_const_iterator;

  inline pred_iterator       pred_begin()      ;
  inline pred_const_iterator pred_begin() const;
  inline pred_iterator       pred_end()        ;
  inline pred_const_iterator pred_end()   const;

  //===--------------------------------------------------------------------===//
  // Successor iterator code
  //===--------------------------------------------------------------------===//
  // 
  // This is used to figure out what basic blocks we could be going to...
  //

  // Forward declare iterator class template...
  template <class _Term, class _BB> class SuccIterator;

  typedef SuccIterator<TerminatorInst*, BasicBlock*> succ_iterator;
  typedef SuccIterator<const TerminatorInst*, 
		       const BasicBlock*> succ_const_iterator;

  inline succ_iterator       succ_begin()      ;
  inline succ_const_iterator succ_begin() const;
  inline succ_iterator       succ_end()        ;
  inline succ_const_iterator succ_end()   const;

  //===--------------------------------------------------------------------===//
  // END of interesting code...
  //===--------------------------------------------------------------------===//
  //
  // Thank god C++ compilers are good at stomping out tons of templated code...
  //
  template <class _Ptr,  class _USE_iterator> // Predecessor Iterator
  class PredIterator {
    const _Ptr ThisBB;
    _USE_iterator It;
  public:
    typedef PredIterator<_Ptr,_USE_iterator> _Self;

    typedef bidirectional_iterator_tag iterator_category;
    typedef _Ptr pointer;

    inline void advancePastConstPool() {
      // Loop to ignore constant pool references
      while (It != ThisBB->use_end() && 
	     ((*It)->getValueType() != Value::InstructionVal))
	++It;
    }

    inline PredIterator(_Ptr BB) : ThisBB(BB), It(BB->use_begin()) {
      advancePastConstPool();
    }
    inline PredIterator(_Ptr BB, bool) : ThisBB(BB), It(BB->use_end()) {}

    inline bool operator==(const _Self& x) const { return It == x.It; }
    inline bool operator!=(const _Self& x) const { return !operator==(x); }

    inline pointer operator*() const { 
      assert ((*It)->getValueType() == Value::InstructionVal);
      return ((Instruction *)(*It))->getParent(); 
    }
    inline pointer *operator->() const { return &(operator*()); }

    inline _Self& operator++() {   // Preincrement
      ++It; advancePastConstPool();
      return *this; 
    }

    inline _Self operator++(int) { // Postincrement
      _Self tmp = *this; ++*this; return tmp; 
    }

    inline _Self& operator--() { --It; return *this; }  // Predecrement
    inline _Self operator--(int) { // Postdecrement
      _Self tmp = *this; --*this; return tmp;
    }
  };

  template <class _Term, class _BB>           // Successor Iterator
  class SuccIterator {
    const _Term Term;
    unsigned idx;
  public:
    typedef SuccIterator<_Term, _BB> _Self;
    typedef forward_iterator_tag iterator_category;
    typedef _BB pointer;
    
    inline SuccIterator(_Term T) : Term(T), idx(0) {}         // begin iterator
    inline SuccIterator(_Term T, bool) 
      : Term(T), idx(Term->getNumSuccessors()) {}             // end iterator
    
    inline bool operator==(const _Self& x) const { return idx == x.idx; }
    inline bool operator!=(const _Self& x) const { return !operator==(x); }

    inline pointer operator*() const { return Term->getSuccessor(idx); }
    inline pointer *operator->() const { return &(operator*()); }
    
    inline _Self& operator++() { ++idx; return *this; } // Preincrement
    inline _Self operator++(int) { // Postincrement
      _Self tmp = *this; ++*this; return tmp; 
    }

    inline _Self& operator--() { --idx; return *this; }  // Predecrement
    inline _Self operator--(int) { // Postdecrement
      _Self tmp = *this; --*this; return tmp;
    }
  };
};


//===--------------------------------------------------------------------===//
// Implement some stuff prototyped above...
//===--------------------------------------------------------------------===//

inline BasicBlock::pred_iterator       BasicBlock::pred_begin()       {
  return pred_iterator(this);
}
inline BasicBlock::pred_const_iterator BasicBlock::pred_begin() const {
  return pred_const_iterator(this);
}
inline BasicBlock::pred_iterator       BasicBlock::pred_end()         {
  return pred_iterator(this,true);
}
inline BasicBlock::pred_const_iterator BasicBlock::pred_end()   const {
  return pred_const_iterator(this,true);
}

inline BasicBlock::succ_iterator       BasicBlock::succ_begin()       {
  return succ_iterator(getTerminator());
}
inline BasicBlock::succ_const_iterator BasicBlock::succ_begin() const {
  return succ_const_iterator(getTerminator());
}
inline BasicBlock::succ_iterator       BasicBlock::succ_end()         {
  return succ_iterator(getTerminator(),true);
}
inline BasicBlock::succ_const_iterator BasicBlock::succ_end()   const {
  return succ_const_iterator(getTerminator(),true);
}

#endif
