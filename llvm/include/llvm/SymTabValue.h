//===-- llvm/SymTabValue.h - Implement SymbolTable Values --------*- C++ -*--=//
//
// This subclass of Value implements a def that has a symbol table for keeping
// track of children.  This is used by the ValueHolder template class...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYMTAB_VALUE_H
#define LLVM_SYMTAB_VALUE_H

class SymbolTable;
class Value;

class SymTabValue {
private:
  SymbolTable *SymTab, *ParentSymTab;
  Value *ValueParent;

protected:
  void setParentSymTab(SymbolTable *ST);
public:
  SymTabValue(Value *Parent);
  ~SymTabValue();    // Implemented in Value.cpp

  inline       Value *getSTVParent()       { return ValueParent; }
  inline const Value *getSTVParent() const { return ValueParent; }

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
  // the method does not already have a symtab, one is created.  Use this if
  // you intend to put something into the symbol table for the method.
  //
  SymbolTable *getSymbolTableSure();  // Implemented in Value.cpp
};

#endif
