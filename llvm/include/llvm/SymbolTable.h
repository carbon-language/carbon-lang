//===-- llvm/SymbolTable.h - Implement a type planed symtab -------*- C++ -*-=//
//
// This file implements a symbol table that has planed broken up by type.  
// Identical types may have overlapping symbol names as long as they are 
// distinct.
//
// Note that this implements a chained symbol table.  If a name being 'lookup'd
// isn't found in the current symbol table, then the parent symbol table is 
// searched.
//
// This chaining behavior does NOT affect iterators though: only the lookup 
// method
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYMBOL_TABLE_H
#define LLVM_SYMBOL_TABLE_H

#include <vector>
#include <map>
#include <string>

class Value;
class Type;

// TODO: Change this back to vector<map<const string, Value *> >
// Make the vector be a data member, and base it on UniqueID's
// That should be much more efficient!
//
class SymbolTable : public map<const Type *, map<const string, Value *> > {
  typedef map<const string, Value *> VarMap;
  typedef map<const Type *, VarMap> super;

  SymbolTable *ParentSymTab;

  friend class SymTabValue;
  inline void setParentSymTab(SymbolTable *P) { ParentSymTab = P; }

public:
  typedef VarMap::iterator type_iterator;
  typedef VarMap::const_iterator type_const_iterator;

  inline SymbolTable(SymbolTable *P = 0) { ParentSymTab = P; }
  ~SymbolTable();

  SymbolTable *getParentSymTab() { return ParentSymTab; }

  // lookup - Returns null on failure...
  Value *lookup(const Type *Ty, const string &name);

  // find - returns end(Ty->getIDNumber()) on failure...
  type_iterator type_find(const Type *Ty, const string &name);
  type_iterator type_find(const Value *D);

  // insert - Add named definition to the symbol table...
  void insert(Value *N);

  void remove(Value *N);
  Value *type_remove(const type_iterator &It);

  inline unsigned type_size(const Type *TypeID) const {
    return find(TypeID)->second.size();
  }

  // Note that type_begin / type_end only work if you know that an element of 
  // TypeID is already in the symbol table!!!
  //
  inline type_iterator type_begin(const Type *TypeID) { 
    return find(TypeID)->second.begin(); 
  }
  inline type_const_iterator type_begin(const Type *TypeID) const {
    return find(TypeID)->second.begin(); 
  }

  inline type_iterator type_end(const Type *TypeID) { 
    return find(TypeID)->second.end(); 
  }
  inline type_const_iterator type_end(const Type *TypeID) const { 
    return find(TypeID)->second.end(); 
  }
};

#endif
