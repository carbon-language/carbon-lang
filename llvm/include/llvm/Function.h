//===-- llvm/Function.h - Class to represent a single VM function -*- C++ -*-=//
//
// This file contains the declaration of the Function class, which represents a 
// single function/procedure in the VM.
//
// Note that BasicBlock's in the Function are Value's, because they are
// referenced by instructions like calls and can go into virtual function tables
// and stuff.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUNCTION_H
#define LLVM_FUNCTION_H

#include "llvm/GlobalValue.h"
#include "llvm/ValueHolder.h"

class FunctionType;

class Function : public GlobalValue {
public:
  typedef ValueHolder<Argument  , Function, Function> ArgumentListType;
  typedef ValueHolder<BasicBlock, Function, Function> BasicBlocksType;

  // BasicBlock iterators...
  typedef BasicBlocksType::iterator iterator;
  typedef BasicBlocksType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

private:

  // Important things that make up a function!
  BasicBlocksType  BasicBlocks;         // The basic blocks
  ArgumentListType ArgumentList;        // The formal arguments

  SymbolTable *SymTab, *ParentSymTab;
  
  friend class ValueHolder<Function, Module, Module>;
  void setParent(Module *parent);

public:
  Function(const FunctionType *Ty, bool isInternal, const std::string &N = "");
  ~Function();

  // Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

  const Type *getReturnType() const;           // Return the type of the ret val
  const FunctionType *getFunctionType() const; // Return the FunctionType for me

  // Is the body of this function unknown? (the basic block list is empty if so)
  // this is true for external functions, defined as forward "declare"ations
  bool isExternal() const { return BasicBlocks.empty(); }

  // Get the underlying elements of the Function... both the argument list and
  // basic block list are empty for external functions.
  //
  inline const ArgumentListType &getArgumentList() const{ return ArgumentList; }
  inline       ArgumentListType &getArgumentList()      { return ArgumentList; }

  inline const BasicBlocksType  &getBasicBlocks() const { return BasicBlocks; }
  inline       BasicBlocksType  &getBasicBlocks()       { return BasicBlocks; }

  inline const BasicBlock       *getEntryNode() const   { return front(); }
  inline       BasicBlock       *getEntryNode()         { return front(); }

  //===--------------------------------------------------------------------===//
  // Symbol Table Accessing functions...

  // hasSymbolTable() - Returns true if there is a symbol table allocated to
  // this object AND if there is at least one name in it!
  //
  bool hasSymbolTable() const;

  // CAUTION: The current symbol table may be null if there are no names (ie, 
  // the symbol table is empty) 
  //
  inline       SymbolTable *getSymbolTable()       { return SymTab; }
  inline const SymbolTable *getSymbolTable() const { return SymTab; }

  // getSymbolTableSure is guaranteed to not return a null pointer, because if
  // the function does not already have a symtab, one is created.  Use this if
  // you intend to put something into the symbol table for the function.
  //
  SymbolTable *getSymbolTableSure();  // Implemented in Value.cpp

  
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

  virtual void print(std::ostream &OS) const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Function *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueType() == Value::FunctionVal;
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
};

#endif
