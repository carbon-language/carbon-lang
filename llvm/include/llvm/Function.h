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
#include "llvm/Support/Annotation.h"

namespace llvm {

class FunctionType;

// Traits for intrusive list of instructions...
template<> struct ilist_traits<BasicBlock>
  : public SymbolTableListTraits<BasicBlock, Function, Function> {

  // createSentinel is used to create a node that marks the end of the list...
  static BasicBlock *createSentinel();
  static void destroySentinel(BasicBlock *BB) { delete BB; }
  static iplist<BasicBlock> &getList(Function *F);
};

template<> struct ilist_traits<Argument>
  : public SymbolTableListTraits<Argument, Function, Function> {

  // createSentinel is used to create a node that marks the end of the list...
  static Argument *createSentinel();
  static void destroySentinel(Argument *A) { delete A; }
  static iplist<Argument> &getList(Function *F);
};

class Function : public GlobalValue, public Annotable {
public:
  typedef iplist<Argument> ArgumentListType;
  typedef iplist<BasicBlock> BasicBlockListType;

  // BasicBlock iterators...
  typedef BasicBlockListType::iterator iterator;
  typedef BasicBlockListType::const_iterator const_iterator;

  typedef ArgumentListType::iterator arg_iterator;
  typedef ArgumentListType::const_iterator const_arg_iterator;

private:
  // Important things that make up a function!
  BasicBlockListType  BasicBlocks;      // The basic blocks
  ArgumentListType ArgumentList;        // The formal arguments

  SymbolTable *SymTab;
  unsigned CallingConvention;

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

  const Type *getReturnType() const;           // Return the type of the ret val
  const FunctionType *getFunctionType() const; // Return the FunctionType for me

  /// isVarArg - Return true if this function takes a variable number of
  /// arguments.
  bool isVarArg() const;

  /// isExternal - Is the body of this function unknown? (The basic block list
  /// is empty if so.) This is true for external functions, defined as forward
  /// "declare"ations
  ///
  virtual bool isExternal() const { return BasicBlocks.empty(); }

  /// getIntrinsicID - This method returns the ID number of the specified
  /// function, or Intrinsic::not_intrinsic if the function is not an
  /// instrinsic, or if the pointer is null.  This value is always defined to be
  /// zero to allow easy checking for whether a function is intrinsic or not.
  /// The particular intrinsic functions which correspond to this value are
  /// defined in llvm/Intrinsics.h.
  ///
  unsigned getIntrinsicID() const;
  bool isIntrinsic() const { return getIntrinsicID() != 0; }

  /// getCallingConv()/setCallingConv(uint) - These method get and set the
  /// calling convention of this function.  The enum values for the known
  /// calling conventions are defined in CallingConv.h.
  unsigned getCallingConv() const { return CallingConvention; }
  void setCallingConv(unsigned CC) { CallingConvention = CC; }

  /// renameLocalSymbols - This method goes through the Function's symbol table
  /// and renames any symbols that conflict with symbols at global scope.  This
  /// is required before printing out to a textual form, to ensure that there is
  /// no ambiguity when parsing.
  void renameLocalSymbols();


  /// deleteBody - This method deletes the body of the function, and converts
  /// the linkage to external.
  ///
  void deleteBody() {
    dropAllReferences();
    setLinkage(ExternalLinkage);
  }

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  void eraseFromParent();


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

  size_t                   size() const { return BasicBlocks.size();  }
  bool                    empty() const { return BasicBlocks.empty(); }
  const BasicBlock       &front() const { return BasicBlocks.front(); }
        BasicBlock       &front()       { return BasicBlocks.front(); }
  const BasicBlock        &back() const { return BasicBlocks.back();  }
        BasicBlock        &back()       { return BasicBlocks.back();  }

  //===--------------------------------------------------------------------===//
  // Argument iterator forwarding functions
  //
  arg_iterator                arg_begin()       { return ArgumentList.begin(); }
  const_arg_iterator          arg_begin() const { return ArgumentList.begin(); }
  arg_iterator                arg_end  ()       { return ArgumentList.end();   }
  const_arg_iterator          arg_end  () const { return ArgumentList.end();   }

  size_t                      arg_size () const { return ArgumentList.size();  }
  bool                        arg_empty() const { return ArgumentList.empty(); }

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
  /// into the nodes, just the label.  If you are only interested in the CFG
  /// this can make the graph smaller.
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
  /// zero.  Then everything is deleted for real.  Note that no operations are
  /// valid on an object that has "dropped all references", except operator
  /// delete.
  ///
  /// Since no other object in the module can have references into the body of a
  /// function, dropping all references deletes the entire body of the function,
  /// including any contained basic blocks.
  ///
  void dropAllReferences();
};

} // End llvm namespace

#endif
