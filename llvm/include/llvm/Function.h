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
#include "llvm/GlobalValue.h"
#include "llvm/ValueHolder.h"

class Instruction;
class BasicBlock;
class MethodArgument;
class MethodType;
class Module;

class Method : public GlobalValue, public SymTabValue {
public:
  typedef ValueHolder<MethodArgument, Method, Method> ArgumentListType;
  typedef ValueHolder<BasicBlock    , Method, Method> BasicBlocksType;

  // BasicBlock iterators...
  typedef BasicBlocksType::iterator iterator;
  typedef BasicBlocksType::const_iterator const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator>             reverse_iterator;

private:

  // Important things that make up a method!
  BasicBlocksType  BasicBlocks;         // The basic blocks
  ArgumentListType ArgumentList;        // The formal arguments
  
  friend class ValueHolder<Method, Module, Module>;
  void setParent(Module *parent);

public:
  Method(const MethodType *Ty, bool isInternal, const std::string &Name = "");
  ~Method();

  // Specialize setName to handle symbol table majik...
  virtual void setName(const std::string &name, SymbolTable *ST = 0);

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
};

typedef Method Function;

#endif
