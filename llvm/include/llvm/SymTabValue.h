//===-- llvm/SymTabDef.h - Implement SymbolTable Defs ------------*- C++ -*--=//
//
// This subclass of Def implements a def that has a symbol table for keeping
// track of children.  This is used by the DefHolder template class...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYMTABDEF_H
#define LLVM_SYMTABDEF_H

#include "llvm/Value.h"               // Get the definition of Value
#include "llvm/ConstantPool.h"

class SymbolTable;
class ConstPoolVal;

class SymTabValue : public Value {
public:
  typedef ConstantPool ConstantPoolType;
private:
  SymbolTable *SymTab, *ParentSymTab;
  ConstantPool ConstPool;   // The constant pool

protected:
  void setParentSymTab(SymbolTable *ST);
public:
  SymTabValue(const Type *Ty, ValueTy dty, const string &name = "");
  ~SymTabValue();    // Implemented in Def.cpp

  // hasSymbolTable() - Returns true if there is a symbol table allocated to
  // this object AND if there is at least one name in it!
  //
  bool hasSymbolTable() const;

  // CAUTION: The current symbol table may be null if there are no names (ie, 
  // the symbol table is empty) 
  //
  inline       SymbolTable *getSymbolTable()       { return SymTab; }
  inline const SymbolTable *getSymbolTable() const { return SymTab; }

  inline const ConstantPool &getConstantPool() const{ return ConstPool; }
  inline       ConstantPool &getConstantPool()      { return ConstPool; }

  // getSymbolTableSure is guaranteed to not return a null pointer, because if
  // the method does not already have a symtab, one is created.  Use this if
  // you intend to put something into the symbol table for the method.
  //
  SymbolTable *getSymbolTableSure();  // Implemented in Def.cpp
};

#endif
