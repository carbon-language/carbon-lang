//===- Core/SymbolTable.h - Main Symbol Table -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_SYMBOL_TABLE_H_
#define LLD_CORE_SYMBOL_TABLE_H_

#include <cstring>
#include <map>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseSet.h"


namespace lld {

class Atom;
class Platform;

/// The SymbolTable class is responsible for coalescing atoms.
///
/// All atoms coalescable by-name or by-content should be added.
/// The method replacement() can be used to find the replacement atom
/// if an atom has been coalesced away.
class SymbolTable {
public:
      SymbolTable(Platform& plat);

  /// @brief add atom to symbol table
  void add(const Atom &);

  /// @brief checks if name is in symbol table and if so atom is not
  ///        UndefinedAtom
  bool isDefined(llvm::StringRef sym);

  /// @brief returns atom in symbol table for specified name (or NULL)
  const Atom *findByName(llvm::StringRef sym);

  /// @brief returns vector of remaining UndefinedAtoms
  void undefines(std::vector<const Atom *>&);

  /// @brief count of by-name entries in symbol table
  unsigned int size();

  /// @brief if atom has been coalesced away, return replacement, else return atom
  const Atom *replacement(const Atom *);

private:
  typedef std::map<llvm::StringRef, const Atom *> NameToAtom;
  typedef std::map<const Atom *, const Atom *> AtomToAtom;
  struct MyMappingInfo {
    static const Atom * getEmptyKey() { return NULL; }
    static const Atom * getTombstoneKey() { return (Atom*)(-1); }
    static unsigned getHashValue(const Atom * const Val);
    static bool isEqual(const Atom * const LHS, const Atom * const RHS);
  };
  typedef llvm::DenseSet<const Atom*, MyMappingInfo> AtomContentSet;

  void addByName(const Atom &);
  void addByContent(const Atom &);

  Platform&  _platform;
  AtomToAtom _replacedAtoms;
  NameToAtom _nameTable;
  AtomContentSet _contentTable;
};

} // namespace lld

#endif // LLD_CORE_SYMBOL_TABLE_H_
