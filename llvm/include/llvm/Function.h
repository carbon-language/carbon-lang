//===-- llvm/Method.h - Class to represent a single VM method ----*- C++ -*--=//
//
// This file contains the declaration of the Method class, which represents a 
// single Method/function/procedure in the VM.
//
// Note that basic blocks themselves are Def's, because they are referenced
// by instructions like calls and can go in virtual function tables and stuff.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_METHOD_H
#define LLVM_METHOD_H

#include "llvm/SymTabValue.h"
#include "llvm/BasicBlock.h"
#include <list>

class Instruction;
class BasicBlock;
class MethodArgument;
class MethodType;
class Method;
class Module;

typedef UseTy<Method> MethodUse;

class Method : public SymTabValue {
public:
  typedef ValueHolder<MethodArgument, Method> ArgumentListType;
  typedef ValueHolder<BasicBlock    , Method> BasicBlocksType;
  typedef BasicBlocksType::iterator iterator;
private:

  // Important things that make up a method!
  BasicBlocksType  BasicBlocks;    // The basic blocks
  ArgumentListType ArgumentList;   // The formal arguments

  Module *Parent;                  // The module that contains this method

  friend class ValueHolder<Method,Module>;
  void setParent(Module *parent);

public:
  Method(const MethodType *Ty, const string &Name = "");
  ~Method();

  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name);

  const Type *getReturnType() const;
  const MethodType *getMethodType() const;

  // Is the body of this method unknown? (the basic block list is empty if so)
  // this is true for "extern"al methods.
  bool isMethodExternal() const { return BasicBlocks.empty(); }


  // Get the class structure that this method is contained inside of...
  inline Module *getParent() { return Parent; }
  inline const Module *getParent() const { return Parent; }

  inline const BasicBlocksType  &getBasicBlocks() const { return BasicBlocks; }
  inline       BasicBlocksType  &getBasicBlocks()       { return BasicBlocks; }

  inline const ArgumentListType &getArgumentList() const{ return ArgumentList; }
  inline       ArgumentListType &getArgumentList()      { return ArgumentList; }


  // dropAllReferences() - This function causes all the subinstructions to "let
  // go" of all references that they are maintaining.  This allows one to
  // 'delete' a whole class at a time, even though there may be circular
  // references... first all references are dropped, and all use counts go to
  // zero.  Then everything is delete'd for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator 
  // delete.
  //
  void dropAllReferences();

  //===--------------------------------------------------------------------===//
  // Method Instruction iterator code
  //===--------------------------------------------------------------------===//
  // 
  template <class _BB_t, class _BB_i_t, class _BI_t, class _II_t> 
  class InstIterator;
  typedef InstIterator<BasicBlocksType, BasicBlocksType::iterator, 
		       BasicBlock::InstListType::iterator,
		       Instruction*> inst_iterator;
  typedef InstIterator<const BasicBlocksType, BasicBlocksType::const_iterator, 
		       BasicBlock::InstListType::const_iterator,
		       const Instruction*> inst_const_iterator;

  // This inner class is used to implement inst_begin() & inst_end() for
  // inst_iterator and inst_const_iterator's.
  //
  template <class _BB_t, class _BB_i_t, class _BI_t, class _II_t>
  class InstIterator {
    typedef _BB_t   BBty;
    typedef _BB_i_t BBIty;
    typedef _BI_t   BIty;
    typedef _II_t   IIty;
    _BB_t  &BBs;      // BasicBlocksType
    _BB_i_t BB;       // BasicBlocksType::iterator
    _BI_t   BI;       // BasicBlock::InstListType::iterator
  public:
    typedef bidirectional_iterator_tag iterator_category;

    template<class M> InstIterator(M &m) 
      : BBs(m.getBasicBlocks()), BB(BBs.begin()) {    // begin ctor
      if (BB != BBs.end()) {
	BI = (*BB)->getInstList().begin();
	resyncInstructionIterator();
      }
    }

    template<class M> InstIterator(M &m, bool) 
      : BBs(m.getBasicBlocks()), BB(BBs.end()) {    // end ctor
    }

    // Accessors to get at the underlying iterators...
    inline BBIty &getBasicBlockIterator()  { return BB; }
    inline BIty  &getInstructionIterator() { return BI; }

    inline IIty operator*()  const { return *BI; }
    inline IIty *operator->() const { return &(operator*()); }

    inline bool operator==(const InstIterator &y) const { 
      return BB == y.BB && (BI == y.BI || BB == BBs.end());
    }
    inline bool operator!=(const InstIterator& y) const { 
      return !operator==(y);
    }

    // resyncInstructionIterator - This should be called if the 
    // InstructionIterator is modified outside of our control.  This resynchs
    // the internals of the InstIterator to a consistent state.
    //
    inline void resyncInstructionIterator() {
      // The only way that the II could be broken is if it is now pointing to
      // the end() of the current BasicBlock and there are successor BBs.
      while (BI == (*BB)->getInstList().end()) {
	++BB; 
	if (BB == BBs.end()) break;
	BI = (*BB)->getInstList().begin();
      }
    }

    InstIterator& operator++() { 
      ++BI;
      resyncInstructionIterator();   // Make sure it is still valid.
      return *this; 
    }
    inline InstIterator operator++(int) { 
      InstIterator tmp = *this; ++*this; return tmp; 
    }
    
    InstIterator& operator--() { 
      while (BB == BBs.end() || BI == (*BB)->getInstList().begin()) {
	--BB;
	BI = (*BB)->getInstList().end();
      }
      --BI;
      return *this; 
    }
    inline InstIterator  operator--(int) { 
      InstIterator tmp = *this; --*this; return tmp; 
    }
  };

  inline inst_iterator inst_begin() { return inst_iterator(*this); }
  inline inst_iterator inst_end()   { return inst_iterator(*this, true); }
  inline inst_const_iterator inst_begin() const { return inst_const_iterator(*this); }
  inline inst_const_iterator inst_end()   const { return inst_const_iterator(*this, true); }
};

#endif
