//===-- llvm/BasicBlock.h - Represent a basic block in the VM ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
// This file contains the declaration of the BasicBlock class, which represents
// a single basic block in the VM.
//
// Note that basic blocks themselves are Value's, because they are referenced
// by instructions like branches and can go in switch tables and stuff...
//
///===---------------------------------------------------------------------===//
//
// Note that well formed basic blocks are formed of a list of instructions 
// followed by a single TerminatorInst instruction.  TerminatorInst's may not
// occur in the middle of basic blocks, and must terminate the blocks.
//
// This code allows malformed basic blocks to occur, because it may be useful
// in the intermediate stage modification to a program.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BASICBLOCK_H
#define LLVM_BASICBLOCK_H

#include "llvm/Instruction.h"
#include "llvm/SymbolTableListTraits.h"
#include "Support/ilist"

namespace llvm {

class TerminatorInst;
template <class Term, class BB> class SuccIterator;  // Successor Iterator
template <class Ptr, class USE_iterator> class PredIterator;

template<> struct ilist_traits<Instruction>
  : public SymbolTableListTraits<Instruction, BasicBlock, Function> {
  // createNode is used to create a node that marks the end of the list...
  static Instruction *createNode();
  static iplist<Instruction> &getList(BasicBlock *BB);
};

struct BasicBlock : public Value {       // Basic blocks are data objects also
  typedef iplist<Instruction> InstListType;
private :
  InstListType InstList;
  BasicBlock *Prev, *Next; // Next and Prev links for our intrusive linked list

  void setParent(Function *parent);
  void setNext(BasicBlock *N) { Next = N; }
  void setPrev(BasicBlock *N) { Prev = N; }
  friend class SymbolTableListTraits<BasicBlock, Function, Function>;

  BasicBlock(const BasicBlock &);     // Do not implement
  void operator=(const BasicBlock &); // Do not implement

public:
  /// Instruction iterators...
  typedef InstListType::iterator                              iterator;
  typedef InstListType::const_iterator                  const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  /// BasicBlock ctor - If the function parameter is specified, the basic block
  /// is automatically inserted at either the end of the function (if
  /// InsertBefore is null), or before the specified basic block.
  ///
  BasicBlock(const std::string &Name = "", Function *Parent = 0,
             BasicBlock *InsertBefore = 0);
  ~BasicBlock();

  // Specialize setName to take care of symbol table majik
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  /// getParent - Return the enclosing method, or null if none
  ///
  const Function *getParent() const { return InstList.getParent(); }
        Function *getParent()       { return InstList.getParent(); }

  // getNext/Prev - Return the next or previous basic block in the list.
        BasicBlock *getNext()       { return Next; }
  const BasicBlock *getNext() const { return Next; }
        BasicBlock *getPrev()       { return Prev; }
  const BasicBlock *getPrev() const { return Prev; }

  /// getTerminator() - If this is a well formed basic block, then this returns
  /// a pointer to the terminator instruction.  If it is not, then you get a
  /// null pointer back.
  ///
  TerminatorInst *getTerminator();
  const TerminatorInst *const getTerminator() const;
  
  //===--------------------------------------------------------------------===//
  /// Instruction iterator methods
  ///
  inline iterator                begin()       { return InstList.begin(); }
  inline const_iterator          begin() const { return InstList.begin(); }
  inline iterator                end  ()       { return InstList.end();   }
  inline const_iterator          end  () const { return InstList.end();   }

  inline reverse_iterator       rbegin()       { return InstList.rbegin(); }
  inline const_reverse_iterator rbegin() const { return InstList.rbegin(); }
  inline reverse_iterator       rend  ()       { return InstList.rend();   }
  inline const_reverse_iterator rend  () const { return InstList.rend();   }

  inline unsigned                 size() const { return InstList.size();  }
  inline bool                    empty() const { return InstList.empty(); }
  inline const Instruction      &front() const { return InstList.front(); }
  inline       Instruction      &front()       { return InstList.front(); }
  inline const Instruction       &back() const { return InstList.back();  }
  inline       Instruction       &back()       { return InstList.back();  }

  /// getInstList() - Return the underlying instruction list container.  You
  /// need to access it directly if you want to modify it currently.
  ///
  const InstListType &getInstList() const { return InstList; }
        InstListType &getInstList()       { return InstList; }

  virtual void print(std::ostream &OS) const { print(OS, 0); }
  void print(std::ostream &OS, AssemblyAnnotationWriter *AAW) const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BasicBlock *BB) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::BasicBlockVal;
  }

  /// dropAllReferences() - This function causes all the subinstructions to "let
  /// go" of all references that they are maintaining.  This allows one to
  /// 'delete' a whole class at a time, even though there may be circular
  /// references... first all references are dropped, and all use counts go to
  /// zero.  Then everything is delete'd for real.  Note that no operations are
  /// valid on an object that has "dropped all references", except operator 
  /// delete.
  ///
  void dropAllReferences();

  /// removePredecessor - This method is used to notify a BasicBlock that the
  /// specified Predecessor of the block is no longer able to reach it.  This is
  /// actually not used to update the Predecessor list, but is actually used to 
  /// update the PHI nodes that reside in the block.  Note that this should be
  /// called while the predecessor still refers to this block.
  ///
  void removePredecessor(BasicBlock *Pred);

  /// splitBasicBlock - This splits a basic block into two at the specified
  /// instruction.  Note that all instructions BEFORE the specified iterator
  /// stay as part of the original basic block, an unconditional branch is added
  /// to the new BB, and the rest of the instructions in the BB are moved to the
  /// new BB, including the old terminator.  The newly formed BasicBlock is
  /// returned.  This function invalidates the specified iterator.
  ///
  /// Note that this only works on well formed basic blocks (must have a 
  /// terminator), and 'I' must not be the end of instruction list (which would
  /// cause a degenerate basic block to be formed, having a terminator inside of
  /// the basic block).
  ///
  BasicBlock *splitBasicBlock(iterator I, const std::string &BBName = "");
};

} // End llvm namespace

#endif
