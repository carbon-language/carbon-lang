//===-- llvm/SymbolTable.h - Implement a type plane'd symtab ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and re-written by Reid
// Spencer. It is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the main symbol table for LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYMBOL_TABLE_H
#define LLVM_SYMBOL_TABLE_H

#include "llvm/Value.h"
#include <map>

namespace llvm {

/// This class provides a symbol table of name/value pairs that is broken
/// up by type. For each Type* there is a "plane" of name/value pairs in
/// the symbol table.  Identical types may have overlapping symbol names as
/// long as they are distinct. The SymbolTable also tracks,  separately, a
/// map of name/type pairs. This allows types to be named. Types are treated
/// distinctly from Values.
///
/// The SymbolTable provides several utility functions for answering common
/// questions about its contents as well as an iterator interface for
/// directly iterating over the contents. To reduce confusion, the terms
/// "type", "value", and "plane" are used consistently. For example,
/// There is a TypeMap typedef that is the mapping of names to Types.
/// Similarly there is a ValueMap typedef that is the mapping of
/// names to Values. Finally, there is a PlaneMap typedef that is the
/// mapping of types to planes of ValueMap. This is the basic structure
/// of the symbol table. When you call type_begin() you're asking
/// for an iterator at the start of the TypeMap. When you call
/// plane_begin(), you're asking for an iterator at the start of
/// the PlaneMap. Finally, when you call value_begin(), you're asking
/// for an iterator at the start of a ValueMap for a specific type
/// plane.
class SymbolTable : public AbstractTypeUser {

/// @name Types
/// @{
public:

  /// @brief A mapping of names to types.
  typedef std::map<const std::string, const Type*> TypeMap;

  /// @brief An iterator over the TypeMap.
  typedef TypeMap::iterator type_iterator;

  /// @brief A const_iterator over the TypeMap.
  typedef TypeMap::const_iterator type_const_iterator;

  /// @brief A mapping of names to values.
  typedef std::map<const std::string, Value *> ValueMap;

  /// @brief An iterator over a ValueMap.
  typedef ValueMap::iterator value_iterator;

  /// @brief A const_iterator over a ValueMap.
  typedef ValueMap::const_iterator value_const_iterator;

  /// @brief A mapping of types to names to values (type planes).
  typedef std::map<const Type *, ValueMap> PlaneMap;

  /// @brief An iterator over the type planes.
  typedef PlaneMap::iterator plane_iterator;

  /// @brief A const_iterator over the type planes
  typedef PlaneMap::const_iterator plane_const_iterator;

/// @}
/// @name Constructors
/// @{
public:

  SymbolTable() : LastUnique(0) {}
  ~SymbolTable();

/// @}
/// @name Accessors
/// @{
public:

  /// This method finds the value with the given \p name in the
  /// type plane \p Ty and returns it. This method will not find any
  /// Types, only Values. Use lookupType to find Types by name.
  /// @returns null on failure, otherwise the Value associated with
  /// the \p name in type plane \p Ty.
  /// @brief Lookup a named, typed value.
  Value *lookup(const Type *Ty, const std::string &name) const;

  /// This method finds the type with the given \p name in the
  /// type  map and returns it.
  /// @returns null if the name is not found, otherwise the Type
  /// associated with the \p name.
  /// @brief Lookup a type by name.
  Type* lookupType(const std::string& name) const;

  /// @returns true iff the type map and the type plane are both not
  /// empty.
  /// @brief Determine if the symbol table is empty
  inline bool isEmpty() const { return pmap.empty() && tmap.empty(); }

  /// @brief The number of name/type pairs is returned.
  inline unsigned num_types() const { return unsigned(tmap.size()); }

  /// Given a base name, return a string that is either equal to it or
  /// derived from it that does not already occur in the symbol table
  /// for the specified type.
  /// @brief Get a name unique to this symbol table
  std::string getUniqueName(const Type *Ty,
                            const std::string &BaseName) const;

  /// This function can be used from the debugger to display the
  /// content of the symbol table while debugging.
  /// @brief Print out symbol table on stderr
  void dump() const;

/// @}
/// @name Iteration
/// @{
public:

  /// Get an iterator that starts at the beginning of the type planes.
  /// The iterator will iterate over the Type/ValueMap pairs in the
  /// type planes.
  inline plane_iterator plane_begin() { return pmap.begin(); }

  /// Get a const_iterator that starts at the beginning of the type
  /// planes.  The iterator will iterate over the Type/ValueMap pairs
  /// in the type planes.
  inline plane_const_iterator plane_begin() const { return pmap.begin(); }

  /// Get an iterator at the end of the type planes. This serves as
  /// the marker for end of iteration over the type planes.
  inline plane_iterator plane_end() { return pmap.end(); }

  /// Get a const_iterator at the end of the type planes. This serves as
  /// the marker for end of iteration over the type planes.
  inline plane_const_iterator plane_end() const { return pmap.end(); }

  /// Get an iterator that starts at the beginning of a type plane.
  /// The iterator will iterate over the name/value pairs in the type plane.
  /// @note The type plane must already exist before using this.
  inline value_iterator value_begin(const Type *Typ) {
    assert(Typ && "Can't get value iterator with null type!");
    return pmap.find(Typ)->second.begin();
  }

  /// Get a const_iterator that starts at the beginning of a type plane.
  /// The iterator will iterate over the name/value pairs in the type plane.
  /// @note The type plane must already exist before using this.
  inline value_const_iterator value_begin(const Type *Typ) const {
    assert(Typ && "Can't get value iterator with null type!");
    return pmap.find(Typ)->second.begin();
  }

  /// Get an iterator to the end of a type plane. This serves as the marker
  /// for end of iteration of the type plane.
  /// @note The type plane must already exist before using this.
  inline value_iterator value_end(const Type *Typ) {
    assert(Typ && "Can't get value iterator with null type!");
    return pmap.find(Typ)->second.end();
  }

  /// Get a const_iterator to the end of a type plane. This serves as the
  /// marker for end of iteration of the type plane.
  /// @note The type plane must already exist before using this.
  inline value_const_iterator value_end(const Type *Typ) const {
    assert(Typ && "Can't get value iterator with null type!");
    return pmap.find(Typ)->second.end();
  }

  /// Get an iterator to the start of the name/Type map.
  inline type_iterator type_begin() { return tmap.begin(); }

  /// @brief Get a const_iterator to the start of the name/Type map.
  inline type_const_iterator type_begin() const { return tmap.begin(); }

  /// Get an iterator to the end of the name/Type map. This serves as the
  /// marker for end of iteration of the types.
  inline type_iterator type_end() { return tmap.end(); }

  /// Get a const-iterator to the end of the name/Type map. This serves
  /// as the marker for end of iteration of the types.
  inline type_const_iterator type_end() const { return tmap.end(); }

  /// This method returns a plane_const_iterator for iteration over
  /// the type planes starting at a specific plane, given by \p Ty.
  /// @brief Find a type plane.
  inline plane_const_iterator find(const Type* Typ) const {
    assert(Typ && "Can't find type plane with null type!");
    return pmap.find(Typ);
  }

  /// This method returns a plane_iterator for iteration over the
  /// type planes starting at a specific plane, given by \p Ty.
  /// @brief Find a type plane.
  inline plane_iterator find(const Type* Typ) {
    assert(Typ && "Can't find type plane with null type!");
    return pmap.find(Typ);
  }


/// @}
/// @name Mutators
/// @{
public:

  /// This method will strip the symbol table of its names leaving the type and
  /// values.
  /// @brief Strip the symbol table.
  bool strip();

  /// Inserts a type into the symbol table with the specified name. There can be
  /// a many-to-one mapping between names and types. This method allows a type
  /// with an existing entry in the symbol table to get a new name.
  /// @brief Insert a type under a new name.
  void insert(const std::string &Name, const Type *Typ);

  /// Remove a type at the specified position in the symbol table.
  /// @returns the removed Type.
  Type* remove(type_iterator TI);

/// @}
/// @name Mutators used by Value::setName and other LLVM internals.
/// @{
public:

  /// This method adds the provided value \p N to the symbol table.  The Value
  /// must have both a name and a type which are extracted and used to place the
  /// value in the correct type plane under the value's name.
  /// @brief Add a named value to the symbol table
  inline void insert(Value *Val) {
    assert(Val && "Can't insert null type into symbol table!");
    assert(Val->hasName() && "Value must be named to go into symbol table!");
    insertEntry(Val->getName(), Val->getType(), Val);
  }

  /// This method removes a named value from the symbol table. The type and name
  /// of the Value are extracted from \p N and used to lookup the Value in the
  /// correct type plane. If the Value is not in the symbol table, this method
  /// silently ignores the request.
  /// @brief Remove a named value from the symbol table.
  void remove(Value* Val);

  /// changeName - Given a value with a non-empty name, remove its existing
  /// entry from the symbol table and insert a new one for Name.  This is
  /// equivalent to doing "remove(V), V->Name = Name, insert(V)", but is faster,
  /// and will not temporarily remove the symbol table plane if V is the last
  /// value in the symtab with that name (which could invalidate iterators to
  /// that plane).
  void changeName(Value *V, const std::string &Name);

/// @}
/// @name Internal Methods
/// @{
private:
  /// @brief Insert a value into the symbol table with the specified name.
  void insertEntry(const std::string &Name, const Type *Ty, Value *V);

  /// This function is called when one of the types in the type plane
  /// is refined.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  /// This function markes a type as being concrete (defined).
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

/// @}
/// @name Internal Data
/// @{
private:

  /// This is the main content of the symbol table. It provides
  /// separate type planes for named values. That is, each named
  /// value is organized into a separate dictionary based on
  /// Type. This means that the same name can be used for different
  /// types without conflict.
  /// @brief The mapping of types to names to values.
  PlaneMap pmap;

  /// This is the type plane. It is separated from the pmap
  /// because the elements of the map are name/Type pairs not
  /// name/Value pairs and Type is not a Value.
  TypeMap tmap;

  /// This value is used to retain the last unique value used
  /// by getUniqueName to generate unique names.
  mutable uint64_t LastUnique;

/// @}

};

} // End llvm namespace

// vim: sw=2

#endif

