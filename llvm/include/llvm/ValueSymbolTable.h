//===-- llvm/ValueSymbolTable.h - Implement a Value Symtab ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer based on the original SymbolTable.h
// written by the LLVM research group and re-written by Reid Spencer.
// It is distributed under the University of Illinois Open Source License. 
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the name/Value symbol table for LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_VALUE_SYMBOL_TABLE_H
#define LLVM_VALUE_SYMBOL_TABLE_H

#include "llvm/Value.h"
#include <map>

namespace llvm {

/// This class provides a symbol table of name/value pairs. It is essentially
/// a std::map<std::string,Value*> but has a controlled interface provided by
/// LLVM as well as ensuring uniqueness of names.
///
class ValueSymbolTable {

/// @name Types
/// @{
public:

  /// @brief A mapping of names to values.
  typedef std::map<const std::string, Value *> ValueMap;

  /// @brief An iterator over a ValueMap.
  typedef ValueMap::iterator iterator;

  /// @brief A const_iterator over a ValueMap.
  typedef ValueMap::const_iterator const_iterator;

/// @}
/// @name Constructors
/// @{
public:

  ValueSymbolTable() : LastUnique(0) {}
  ~ValueSymbolTable();

/// @}
/// @name Accessors
/// @{
public:

  /// This method finds the value with the given \p name in the
  /// the symbol table. 
  /// @returns the value associated with the \p name
  /// @brief Lookup a named Value.
  Value *lookup(const std::string &name) const;

  /// @returns true iff the symbol table is empty
  /// @brief Determine if the symbol table is empty
  inline bool empty() const { return vmap.empty(); }

  /// @brief The number of name/type pairs is returned.
  inline unsigned size() const { return unsigned(vmap.size()); }

  /// Given a base name, return a string that is either equal to it or
  /// derived from it that does not already occur in the symbol table
  /// for the specified type.
  /// @brief Get a name unique to this symbol table
  std::string getUniqueName(const std::string &BaseName) const;

  /// This function can be used from the debugger to display the
  /// content of the symbol table while debugging.
  /// @brief Print out symbol table on stderr
  void dump() const;

/// @}
/// @name Iteration
/// @{
public:

  /// @brief Get an iterator that from the beginning of the symbol table.
  inline iterator begin() { return vmap.begin(); }

  /// @brief Get a const_iterator that from the beginning of the symbol table.
  inline const_iterator begin() const { return vmap.begin(); }

  /// @brief Get an iterator to the end of the symbol table.
  inline iterator end() { return vmap.end(); }

  /// @brief Get a const_iterator to the end of the symbol table.
  inline const_iterator end() const { return vmap.end(); }

/// @}
/// @name Mutators
/// @{
public:

  /// This method will strip the symbol table of its names.
  /// @brief Strip the symbol table.
  bool strip();

  /// This method adds the provided value \p N to the symbol table.  The Value
  /// must have a name which is used to place the value in the symbol table. 
  /// @brief Add a named value to the symbol table
  void insert(Value *Val);

  /// This method removes a value from the symbol table. The name of the
  /// Value is extracted from \p Val and used to lookup the Value in the
  /// symbol table. If the Value is not in the symbol table, this method
  /// returns false.
  /// @returns true if \p Val was successfully erased, false otherwise
  /// @brief Remove a value from the symbol table.
  bool erase(Value* Val);

  /// Given a value with a non-empty name, remove its existing
  /// entry from the symbol table and insert a new one for Name.  This is
  /// equivalent to doing "remove(V), V->Name = Name, insert(V)".
  /// @brief Rename a value in the symbol table
  bool rename(Value *V, const std::string &Name);

/// @}
/// @name Internal Data
/// @{
private:
  ValueMap vmap;                    ///< The map that holds the symbol table.
  mutable unsigned long LastUnique; ///< Counter for tracking unique names

/// @}

};

} // End llvm namespace

#endif
