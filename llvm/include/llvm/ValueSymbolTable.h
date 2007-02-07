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
  template<typename ValueSubClass, typename ItemParentClass,
           typename SymTabClass, typename SubClass>
        class SymbolTableListTraits;
  template<typename NodeTy> struct ilist_traits;
  class BasicBlock;
  class Function;
  class Module;
  
/// This class provides a symbol table of name/value pairs. It is essentially
/// a std::map<std::string,Value*> but has a controlled interface provided by
/// LLVM as well as ensuring uniqueness of names.
///
class ValueSymbolTable {
  friend class Value;
  friend class SymbolTableListTraits<Argument, Function, Function,
                                     ilist_traits<Argument> >;
  friend class SymbolTableListTraits<BasicBlock, Function, Function,
                                     ilist_traits<BasicBlock> >;
  friend class SymbolTableListTraits<Instruction, BasicBlock, Function,
                                     ilist_traits<Instruction> >;
  friend class SymbolTableListTraits<Function, Module, Module, 
                                     ilist_traits<Function> >;
  friend class SymbolTableListTraits<GlobalVariable, Module, Module, 
                                     ilist_traits<GlobalVariable> >;
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

  /// @return 1 if the name is in the symbol table, 0 otherwise
  /// @brief Determine if a name is in the symbol table
  bool count(const std::string &name) const { 
    return vmap.count(name);
  }

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
private:
  /// This method adds the provided value \p N to the symbol table.  The Value
  /// must have a name which is used to place the value in the symbol table. 
  /// @brief Add a named value to the symbol table
  void insert(Value *Val);

  /// This method removes a value from the symbol table. The name of the
  /// Value is extracted from \p Val and used to lookup the Value in the
  /// symbol table.  \p Val is not deleted, just removed from the symbol table.
  /// @brief Remove a value from the symbol table.
  void remove(Value* Val);
  
/// @}
/// @name Internal Data
/// @{
private:
  ValueMap vmap;                    ///< The map that holds the symbol table.
  mutable uint32_t LastUnique; ///< Counter for tracking unique names

/// @}
};

} // End llvm namespace

#endif
