//===-- llvm/SymbolTable.h - Implement a type plane'd symtab ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a symbol table that has planes broken up by type.  
// Identical types may have overlapping symbol names as long as they are 
// distinct.
//
// Note that this implements a chained symbol table.  If a name being 'lookup'd
// isn't found in the current symbol table, then the parent symbol table is 
// searched.
//
// This chaining behavior does NOT affect iterators though: only the lookup 
// method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYMBOL_TABLE_H
#define LLVM_SYMBOL_TABLE_H

#include "llvm/Value.h"
#include <map>

class SymbolTable : public AbstractTypeUser,
		    public std::map<const Type *, 
                                    std::map<const std::string, Value *> > {
public:
  typedef std::map<const std::string, Value *> VarMap;
  typedef std::map<const Type *, VarMap> super;

  typedef VarMap::iterator type_iterator;
  typedef VarMap::const_iterator type_const_iterator;

  inline SymbolTable() : InternallyInconsistent(false), LastUnique(0) {}
  ~SymbolTable();

  // lookup - Returns null on failure...
  Value *lookup(const Type *Ty, const std::string &name);

  // insert - Add named definition to the symbol table...
  inline void insert(Value *N) {
    assert(N->hasName() && "Value must be named to go into symbol table!");
    insertEntry(N->getName(), N->getType(), N);
  }

  void remove(Value *N);
  Value *type_remove(const type_iterator &It) {
    return removeEntry(find(It->second->getType()), It);
  }

  // insert - Insert a constant or type into the symbol table with the specified
  // name...  There can be a many to one mapping between names and
  // (constant/type)s.
  //
  inline void insert(const std::string &Name, Value *V) {
    assert((isa<Type>(V) || isa<Constant>(V)) &&
	   "Can only insert types and constants here!");
    insertEntry(Name, V->getType(), V);
  }

  /// remove - Remove a constant or type from the symbol table with the
  /// specified name.
  Value *remove(const std::string &Name, Value *V) {
    iterator TI = find(V->getType());
    return removeEntry(TI, TI->second.find(Name));
  }

  // getUniqueName - Given a base name, return a string that is either equal to
  // it (or derived from it) that does not already occur in the symbol table for
  // the specified type.
  //
  std::string getUniqueName(const Type *Ty, const std::string &BaseName);

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

  void dump() const;  // Debug method, print out symbol table

private:
  // InternallyInconsistent - There are times when the symbol table is
  // internally inconsistent with the rest of the program.  In this one case, a
  // value exists with a Name, and it's not in the symbol table.  When we call
  // V->setName(""), it tries to remove itself from the symbol table and dies.
  // We know this is happening, and so if the flag InternallyInconsistent is
  // set, removal from the symbol table is a noop.
  //
  bool InternallyInconsistent;

  // LastUnique - This value is used to retain the last unique value used
  // by getUniqueName to generate unique names.
  unsigned long LastUnique;

  inline super::value_type operator[](const Type *Ty) {
    assert(0 && "Should not use this operator to access symbol table!");
    return super::value_type();
  }

  // insertEntry - Insert a value into the symbol table with the specified
  // name...
  //
  void insertEntry(const std::string &Name, const Type *Ty, Value *V);

  // removeEntry - Remove a value from the symbol table...
  //
  Value *removeEntry(iterator Plane, type_iterator Entry);

  // This function is called when one of the types in the type plane are refined
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);
};

#endif
