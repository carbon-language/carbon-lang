//===-- llvm/Function.h - Class to represent a single function --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Function class, which represents a 
// single function/procedure in LLVM.
//
// A function basically consists of a list of basic blocks, a list of arguments,
// and a symbol table.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUNCTION_H
#define LLVM_FUNCTION_H

#include "llvm/GlobalValue.h"
#include "llvm/BasicBlock.h"
#include "llvm/Argument.h"

class FunctionType;

// Traits for intrusive list of instructions...
template<> struct ilist_traits<BasicBlock>
  : public SymbolTableListTraits<BasicBlock, Function, Function> {

  // createNode is used to create a node that marks the end of the list...
  static BasicBlock *createNode();

  static iplist<BasicBlock> &getList(Function *F);
};

template<> struct ilist_traits<Argument>
  : public SymbolTableListTraits<Argument, Function, Function> {

  // createNode is used to create a node that marks the end of the list...
  static Argument *createNode();
  static iplist<Argument> &getList(Function *F);
};

class Function : public GlobalValue {
public:
  typedef iplist<Argument> ArgumentListType;
  typedef iplist<BasicBlock> BasicBlockListType;

  // BasicBlock iterators...
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

  typedef ArgumentListType::iterator aiterator;
  typedef ArgumentListType::const_iterator const_aiterator;
  typedef std::reverse_iterator<const_aiterator> const_reverse_aiterator;
  typedef std::reverse_iterator<aiterator>             reverse_aiterator;

private:

  // Important things that make up a function!
  BasicBlockListType  BasicBlocks;      // The basic blocks
  ArgumentListType ArgumentList;        // The formal arguments

  SymbolTable *SymTab;
  
  friend class SymbolTableListTraits<Function, Module, Module>;

  void setParent(Module *parent);
  Function *Prev, *Next;
  void setNext(Function *N) { Next = N; }
  void setPrev(Function *N) { Prev = N; }

public:
  /// Function ctor - If the (optional) Module argument is specified, the
  /// function is automatically inserted into the end of the function list for
  /// the module.
  ///
  Function(const FunctionType *Ty, LinkageTypes Linkage,
           const std::string &N = "", Module *M = 0);
  ~Function();

  // Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  const Type *getReturnType() const;           // Return the type of the ret val
  const FunctionType *getFunctionType() const; // Return the FunctionType for me

  /// isExternal - Is the body of this function unknown? (the basic block list
  /// is empty if so) this is true for external functions, defined as forward
  /// "declare"ations
  ///
  virtual bool isExternal() const { return BasicBlocks.empty(); }

  /// getIntrinsicID - This method returns the ID number of the specified
  /// function, or LLVMIntrinsic::not_intrinsic if the function is not an
  /// instrinsic, or if the pointer is null.  This value is always defined to be
  /// zero to allow easy checking for whether a function is intrinsic or not.
  /// The particular intrinsic functions which correspond to this value are
  /// defined in llvm/Intrinsics.h.
  ///
  unsigned getIntrinsicID() const;
  bool isIntrinsic() const { return getIntrinsicID() != 0; }

  /// deleteBody - This method deletes the body of the function, and converts
  /// the linkage to external.
  void deleteBody() {
    dropAllReferences();
    setLinkage(ExternalLinkage);
  }

  // getNext/Prev - Return the next or previous function in the list.  These
  // methods should never be used directly, and are only used to implement the
  // function list as part of the module.
  //
        Function *getNext()       { return Next; }
  const Function *getNext() const { return Next; }
        Function *getPrev()       { return Prev; }
  const Function *getPrev() const { return Prev; }

  /// Get the underlying elements of the Function... the basic block list is
  /// empty for external functions.
  ///
  const ArgumentListType &getArgumentList() const { return ArgumentList; }
        ArgumentListType &getArgumentList()       { return ArgumentList; }

  const BasicBlockListType &getBasicBlockList() const { return BasicBlocks; }
        BasicBlockListType &getBasicBlockList()       { return BasicBlocks; }

  const BasicBlock       &getEntryBlock() const   { return front(); }
        BasicBlock       &getEntryBlock()         { return front(); }

  //===--------------------------------------------------------------------===//
  // Symbol Table Accessing functions...

  /// getSymbolTable() - Return the symbol table...
  ///
  inline       SymbolTable &getSymbolTable()       { return *SymTab; }
  inline const SymbolTable &getSymbolTable() const { return *SymTab; }

  
  //===--------------------------------------------------------------------===//
  // BasicBlock iterator forwarding functions
  //
  iterator                begin()       { return BasicBlocks.begin(); }
  const_iterator          begin() const { return BasicBlocks.begin(); }
  iterator                end  ()       { return BasicBlocks.end();   }
  const_iterator          end  () const { return BasicBlocks.end();   }

  reverse_iterator       rbegin()       { return BasicBlocks.rbegin(); }
  const_reverse_iterator rbegin() const { return BasicBlocks.rbegin(); }
  reverse_iterator       rend  ()       { return BasicBlocks.rend();   }
  const_reverse_iterator rend  () const { return BasicBlocks.rend();   }

  unsigned                 size() const { return BasicBlocks.size(); }
  bool                    empty() const { return BasicBlocks.empty(); }
  const BasicBlock       &front() const { return BasicBlocks.front(); }
        BasicBlock       &front()       { return BasicBlocks.front(); }
  const BasicBlock        &back() const { return BasicBlocks.back(); }
        BasicBlock        &back()       { return BasicBlocks.back(); }

  //===--------------------------------------------------------------------===//
  // Argument iterator forwarding functions
  //
  aiterator                abegin()       { return ArgumentList.begin(); }
  const_aiterator          abegin() const { return ArgumentList.begin(); }
  aiterator                aend  ()       { return ArgumentList.end();   }
  const_aiterator          aend  () const { return ArgumentList.end();   }

  reverse_aiterator       arbegin()       { return ArgumentList.rbegin(); }
  const_reverse_aiterator arbegin() const { return ArgumentList.rbegin(); }
  reverse_aiterator       arend  ()       { return ArgumentList.rend();   }
  const_reverse_aiterator arend  () const { return ArgumentList.rend();   }

  unsigned                  asize() const { return ArgumentList.size(); }
  bool                     aempty() const { return ArgumentList.empty(); }
  const Argument          &afront() const { return ArgumentList.front(); }
        Argument          &afront()       { return ArgumentList.front(); }
  const Argument           &aback() const { return ArgumentList.back(); }
        Argument           &aback()       { return ArgumentList.back(); }

  virtual void print(std::ostream &OS) const { print(OS, 0); }
  void print(std::ostream &OS, AssemblyAnnotationWriter *AAW) const;

  /// viewCFG - This function is meant for use from the debugger.  You can just
  /// say 'call F->viewCFG()' and a ghostview window should pop up from the
  /// program, displaying the CFG of the current function with the code for each
  /// basic block inside.  This depends on there being a 'dot' and 'gv' program
  /// in your path.
  ///
  void viewCFG() const;
  
  /// viewCFGOnly - This function is meant for use from the debugger.  It works
  /// just like viewCFG, but it does not include the contents of basic blocks
  /// into the nodes, just the label.  If you are only interested in the CFG t
  /// his can make the graph smaller.
  ///
  void viewCFGOnly() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Function *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::FunctionVal;
  }

  /// dropAllReferences() - This method causes all the subinstructions to "let
  /// go" of all references that they are maintaining.  This allows one to
  /// 'delete' a whole module at a time, even though there may be circular
  /// references... first all references are dropped, and all use counts go to
  /// zero.  Then everything is delete'd for real.  Note that no operations are
  /// valid on an object that has "dropped all references", except operator 
  /// delete.
  ///
  /// Since no other object in the module can have references into the body of a
  /// function, dropping all references deletes the entire body of the function,
  /// including any contained basic blocks.
  ///
  void dropAllReferences();
};

#endif
