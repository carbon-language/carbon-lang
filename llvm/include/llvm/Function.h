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
#include "llvm/GlobalValue.h"

class Instruction;
class BasicBlock;
class MethodArgument;
class MethodType;
class Module;
class MachineCodeForMethod;

class Method : public GlobalValue, public SymTabValue {
public:
  typedef ValueHolder<MethodArgument, Method, Method> ArgumentListType;
  typedef ValueHolder<BasicBlock    , Method, Method> BasicBlocksType;

  // BasicBlock iterators...
  typedef BasicBlocksType::iterator iterator;
  typedef BasicBlocksType::const_iterator const_iterator;
  typedef reverse_iterator<const_iterator> const_reverse_iterator;
  typedef reverse_iterator<iterator>             reverse_iterator;

private:

  // Important things that make up a method!
  BasicBlocksType  BasicBlocks;         // The basic blocks
  ArgumentListType ArgumentList;        // The formal arguments
  MachineCodeForMethod* machineCode;    // Access to the generated native code
  
  friend class ValueHolder<Method, Module, Module>;
  void setParent(Module *parent);

public:
  Method(const MethodType *Ty, const string &Name = "");
  ~Method();

  // Specialize setName to handle symbol table majik...
  virtual void setName(const string &name, SymbolTable *ST = 0);

  const Type *getReturnType() const;        // Return the return type of method
  const MethodType *getMethodType() const;  // Return the MethodType for me

  // Is the body of this method unknown? (the basic block list is empty if so)
  // this is true for external methods, defined as forward "declare"ations
  bool isExternal() const { return BasicBlocks.empty(); }

  // Get the underlying elements of the Method... both the argument list and
  // basic block list are empty for external methods.
  //
  inline const ArgumentListType &getArgumentList() const{ return ArgumentList; }
  inline       ArgumentListType &getArgumentList()      { return ArgumentList; }

  inline const BasicBlocksType  &getBasicBlocks() const { return BasicBlocks; }
  inline       BasicBlocksType  &getBasicBlocks()       { return BasicBlocks; }

  inline const BasicBlock       *getEntryNode() const   { return front(); }
  inline       BasicBlock       *getEntryNode()         { return front(); }
  
  inline const MachineCodeForMethod
                                &getMachineCode() const { return *machineCode;}
  inline       MachineCodeForMethod
                                &getMachineCode()       { return *machineCode;}
  
  //===--------------------------------------------------------------------===//
  // BasicBlock iterator forwarding functions
  //
  inline iterator                begin()       { return BasicBlocks.begin(); }
  inline const_iterator          begin() const { return BasicBlocks.begin(); }
  inline iterator                end  ()       { return BasicBlocks.end();   }
  inline const_iterator          end  () const { return BasicBlocks.end();   }

  inline reverse_iterator       rbegin()       { return BasicBlocks.rbegin(); }
  inline const_reverse_iterator rbegin() const { return BasicBlocks.rbegin(); }
  inline reverse_iterator       rend  ()       { return BasicBlocks.rend();   }
  inline const_reverse_iterator rend  () const { return BasicBlocks.rend();   }

  inline unsigned                 size() const { return BasicBlocks.size(); }
  inline bool                    empty() const { return BasicBlocks.empty(); }
  inline const BasicBlock       *front() const { return BasicBlocks.front(); }
  inline       BasicBlock       *front()       { return BasicBlocks.front(); }
  inline const BasicBlock        *back() const { return BasicBlocks.back(); }
  inline       BasicBlock        *back()       { return BasicBlocks.back(); }


  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Method *T) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::MethodVal;
  }

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
  typedef InstIterator<BasicBlocksType, iterator, 
		       BasicBlock::iterator, Instruction*> inst_iterator;
  typedef InstIterator<const BasicBlocksType, const_iterator, 
		       BasicBlock::const_iterator,
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
    _BI_t   BI;       // BasicBlock::iterator
  public:
    typedef bidirectional_iterator_tag iterator_category;

    template<class M> InstIterator(M &m) 
      : BBs(m.getBasicBlocks()), BB(BBs.begin()) {    // begin ctor
      if (BB != BBs.end()) {
	BI = (*BB)->begin();
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
    inline IIty operator->() const { return operator*(); }

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
      while (BI == (*BB)->end()) {
	++BB;
	if (BB == BBs.end()) break;
	BI = (*BB)->begin();
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
      while (BB == BBs.end() || BI == (*BB)->begin()) {
	--BB;
	BI = (*BB)->end();
      }
      --BI;
      return *this; 
    }
    inline InstIterator  operator--(int) { 
      InstIterator tmp = *this; --*this; return tmp; 
    }

    inline bool atEnd() const { return BB == BBs.end(); }
  };

  inline inst_iterator inst_begin() { return inst_iterator(*this); }
  inline inst_iterator inst_end()   { return inst_iterator(*this, true); }
  inline inst_const_iterator inst_begin() const { return inst_const_iterator(*this); }
  inline inst_const_iterator inst_end()   const { return inst_const_iterator(*this, true); }
};

// Provide specializations of GraphTraits to be able to treat a method as a 
// graph of basic blocks... these are the same as the basic block iterators,
// except that the root node is implicitly the first node of the method.
//
template <> struct GraphTraits<Method*> : public GraphTraits<BasicBlock*> {
  static NodeType *getEntryNode(Method *M) { return M->front(); }
};
template <> struct GraphTraits<const Method*> :
  public GraphTraits<const BasicBlock*> {
  static NodeType *getEntryNode(const Method *M) { return M->front(); }
};

// Provide specializations of GraphTraits to be able to treat a method as a 
// graph of basic blocks... and to walk it in inverse order.  Inverse order for
// a method is considered to be when traversing the predecessor edges of a BB
// instead of the successor edges.
//
template <> struct GraphTraits<Inverse<Method*> > :
  public GraphTraits<Inverse<BasicBlock*> > {
  static NodeType *getEntryNode(Inverse<Method *> G) { return G.Graph->front();}
};
template <> struct GraphTraits<Inverse<const Method*> > :
  public GraphTraits<Inverse<const BasicBlock*> > {
  static NodeType *getEntryNode(Inverse<const Method *> G) {
    return G.Graph->front();
  }
};

#endif
