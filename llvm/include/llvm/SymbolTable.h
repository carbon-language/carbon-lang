//===-- llvm/SymbolTable.h - Implement a type planned symtab ------*- C++ -*-=//
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

#include "llvm/Value.h"
#include <map>

#ifndef NDEBUG             // Only for assertions
#include "llvm/Type.h"
#include "llvm/ConstPoolVals.h"
#endif

class Value;
class Type;

// TODO: Change this back to vector<map<const string, Value *> >
// Make the vector be a data member, and base it on UniqueID's
// That should be much more efficient!
//
class SymbolTable : public AbstractTypeUser,
		    public map<const Type *, map<const string, Value *> > {
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
  inline void insert(Value *N) {
    assert(N->hasName() && "Value must be named to go into symbol table!");
    insertEntry(N->getName(), N);
  }

  // insert - Insert a constant or type into the symbol table with the specified
  // name...  There can be a many to one mapping between names and
  // (constant/type)s.
  //
  inline void insert(const string &Name, Value *V) {
    assert((isa<Type>(V) || isa<ConstPoolVal>(V)) &&
	   "Can only insert types and constants here!");
    insertEntry(Name, V);
  }

  void remove(Value *N);
  Value *type_remove(const type_iterator &It);

  // getUniqueName - Given a base name, return a string that is either equal to
  // it (or derived from it) that does not already occur in the symbol table for
  // the specified type.
  //
  string getUniqueName(const Type *Ty, const string &BaseName);

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

private:
  // insertEntry - Insert a value into the symbol table with the specified
  // name...
  //
  void insertEntry(const string &Name, Value *V);

  // This function is called when one of the types in the type plane are refined
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
};

#endif
