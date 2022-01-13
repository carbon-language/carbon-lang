//===-- Symtab.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_SYMTAB_H
#define LLDB_SYMBOL_SYMTAB_H

#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Utility/RangeMap.h"
#include "lldb/lldb-private.h"
#include <map>
#include <mutex>
#include <vector>

namespace lldb_private {

class Symtab {
public:
  typedef std::vector<uint32_t> IndexCollection;
  typedef UniqueCStringMap<uint32_t> NameToIndexMap;

  enum Debug {
    eDebugNo,  // Not a debug symbol
    eDebugYes, // A debug symbol
    eDebugAny
  };

  enum Visibility { eVisibilityAny, eVisibilityExtern, eVisibilityPrivate };

  Symtab(ObjectFile *objfile);
  ~Symtab();

  void PreloadSymbols();
  void Reserve(size_t count);
  Symbol *Resize(size_t count);
  uint32_t AddSymbol(const Symbol &symbol);
  size_t GetNumSymbols() const;
  void SectionFileAddressesChanged();
  void
  Dump(Stream *s, Target *target, SortOrder sort_type,
       Mangled::NamePreference name_preference = Mangled::ePreferDemangled);
  void Dump(Stream *s, Target *target, std::vector<uint32_t> &indexes,
            Mangled::NamePreference name_preference =
                Mangled::ePreferDemangled) const;
  uint32_t GetIndexForSymbol(const Symbol *symbol) const;
  std::recursive_mutex &GetMutex() { return m_mutex; }
  Symbol *FindSymbolByID(lldb::user_id_t uid) const;
  Symbol *SymbolAtIndex(size_t idx);
  const Symbol *SymbolAtIndex(size_t idx) const;
  Symbol *FindSymbolWithType(lldb::SymbolType symbol_type,
                             Debug symbol_debug_type,
                             Visibility symbol_visibility, uint32_t &start_idx);
  /// Get the parent symbol for the given symbol.
  ///
  /// Many symbols in symbol tables are scoped by other symbols that
  /// contain one or more symbol. This function will look for such a
  /// containing symbol and return it if there is one.
  const Symbol *GetParent(Symbol *symbol) const;
  uint32_t AppendSymbolIndexesWithType(lldb::SymbolType symbol_type,
                                       std::vector<uint32_t> &indexes,
                                       uint32_t start_idx = 0,
                                       uint32_t end_index = UINT32_MAX) const;
  uint32_t AppendSymbolIndexesWithTypeAndFlagsValue(
      lldb::SymbolType symbol_type, uint32_t flags_value,
      std::vector<uint32_t> &indexes, uint32_t start_idx = 0,
      uint32_t end_index = UINT32_MAX) const;
  uint32_t AppendSymbolIndexesWithType(lldb::SymbolType symbol_type,
                                       Debug symbol_debug_type,
                                       Visibility symbol_visibility,
                                       std::vector<uint32_t> &matches,
                                       uint32_t start_idx = 0,
                                       uint32_t end_index = UINT32_MAX) const;
  uint32_t AppendSymbolIndexesWithName(ConstString symbol_name,
                                       std::vector<uint32_t> &matches);
  uint32_t AppendSymbolIndexesWithName(ConstString symbol_name,
                                       Debug symbol_debug_type,
                                       Visibility symbol_visibility,
                                       std::vector<uint32_t> &matches);
  uint32_t AppendSymbolIndexesWithNameAndType(ConstString symbol_name,
                                              lldb::SymbolType symbol_type,
                                              std::vector<uint32_t> &matches);
  uint32_t AppendSymbolIndexesWithNameAndType(ConstString symbol_name,
                                              lldb::SymbolType symbol_type,
                                              Debug symbol_debug_type,
                                              Visibility symbol_visibility,
                                              std::vector<uint32_t> &matches);
  uint32_t
  AppendSymbolIndexesMatchingRegExAndType(const RegularExpression &regex,
                                          lldb::SymbolType symbol_type,
                                          std::vector<uint32_t> &indexes);
  uint32_t AppendSymbolIndexesMatchingRegExAndType(
      const RegularExpression &regex, lldb::SymbolType symbol_type,
      Debug symbol_debug_type, Visibility symbol_visibility,
      std::vector<uint32_t> &indexes);
  void FindAllSymbolsWithNameAndType(ConstString name,
                                     lldb::SymbolType symbol_type,
                                     std::vector<uint32_t> &symbol_indexes);
  void FindAllSymbolsWithNameAndType(ConstString name,
                                     lldb::SymbolType symbol_type,
                                     Debug symbol_debug_type,
                                     Visibility symbol_visibility,
                                     std::vector<uint32_t> &symbol_indexes);
  void FindAllSymbolsMatchingRexExAndType(
      const RegularExpression &regex, lldb::SymbolType symbol_type,
      Debug symbol_debug_type, Visibility symbol_visibility,
      std::vector<uint32_t> &symbol_indexes);
  Symbol *FindFirstSymbolWithNameAndType(ConstString name,
                                         lldb::SymbolType symbol_type,
                                         Debug symbol_debug_type,
                                         Visibility symbol_visibility);
  Symbol *FindSymbolAtFileAddress(lldb::addr_t file_addr);
  Symbol *FindSymbolContainingFileAddress(lldb::addr_t file_addr);
  void ForEachSymbolContainingFileAddress(
      lldb::addr_t file_addr, std::function<bool(Symbol *)> const &callback);
  void FindFunctionSymbols(ConstString name, uint32_t name_type_mask,
                           SymbolContextList &sc_list);
  void CalculateSymbolSizes();

  void SortSymbolIndexesByValue(std::vector<uint32_t> &indexes,
                                bool remove_duplicates) const;

  static void DumpSymbolHeader(Stream *s);

  void Finalize() {
    // Shrink to fit the symbols so we don't waste memory
    if (m_symbols.capacity() > m_symbols.size()) {
      collection new_symbols(m_symbols.begin(), m_symbols.end());
      m_symbols.swap(new_symbols);
    }
  }

  void AppendSymbolNamesToMap(const IndexCollection &indexes,
                              bool add_demangled, bool add_mangled,
                              NameToIndexMap &name_to_index_map) const;

  ObjectFile *GetObjectFile() { return m_objfile; }

protected:
  typedef std::vector<Symbol> collection;
  typedef collection::iterator iterator;
  typedef collection::const_iterator const_iterator;
  class FileRangeToIndexMapCompare {
  public:
    FileRangeToIndexMapCompare(const Symtab &symtab) : m_symtab(symtab) {}
    bool operator()(const uint32_t a_data, const uint32_t b_data) const {
      return rank(a_data) > rank(b_data);
    }

  private:
    // How much preferred is this symbol?
    int rank(const uint32_t data) const {
      const Symbol &symbol = *m_symtab.SymbolAtIndex(data);
      if (symbol.IsExternal())
        return 3;
      if (symbol.IsWeak())
        return 2;
      if (symbol.IsDebug())
        return 0;
      return 1;
    }
    const Symtab &m_symtab;
  };
  typedef RangeDataVector<lldb::addr_t, lldb::addr_t, uint32_t, 0,
                          FileRangeToIndexMapCompare>
      FileRangeToIndexMap;
  void InitNameIndexes();
  void InitAddressIndexes();

  ObjectFile *m_objfile;
  collection m_symbols;
  FileRangeToIndexMap m_file_addr_to_index;

  /// Maps function names to symbol indices (grouped by FunctionNameTypes)
  std::map<lldb::FunctionNameType, UniqueCStringMap<uint32_t>>
      m_name_to_symbol_indices;
  mutable std::recursive_mutex
      m_mutex; // Provide thread safety for this symbol table
  bool m_file_addr_to_index_computed : 1, m_name_indexes_computed : 1;

private:
  UniqueCStringMap<uint32_t> &
  GetNameToSymbolIndexMap(lldb::FunctionNameType type) {
    auto map = m_name_to_symbol_indices.find(type);
    assert(map != m_name_to_symbol_indices.end());
    return map->second;
  }
  bool CheckSymbolAtIndex(size_t idx, Debug symbol_debug_type,
                          Visibility symbol_visibility) const {
    switch (symbol_debug_type) {
    case eDebugNo:
      if (m_symbols[idx].IsDebug())
        return false;
      break;

    case eDebugYes:
      if (!m_symbols[idx].IsDebug())
        return false;
      break;

    case eDebugAny:
      break;
    }

    switch (symbol_visibility) {
    case eVisibilityAny:
      return true;

    case eVisibilityExtern:
      return m_symbols[idx].IsExternal();

    case eVisibilityPrivate:
      return !m_symbols[idx].IsExternal();
    }
    return false;
  }

  /// A helper function that looks up full function names.
  ///
  /// We generate unique names for synthetic symbols so that users can look
  /// them up by name when needed. But because doing so is uncommon in normal
  /// debugger use, we trade off some performance at lookup time for faster
  /// symbol table building by detecting these symbols and generating their
  /// names lazily, rather than adding them to the normal symbol indexes. This
  /// function does the job of first consulting the name indexes, and if that
  /// fails it extracts the information it needs from the synthetic name and
  /// locates the symbol.
  ///
  /// @param[in] symbol_name The symbol name to search for.
  ///
  /// @param[out] indexes The vector if symbol indexes to update with results.
  ///
  /// @returns The number of indexes added to the index vector. Zero if no
  /// matches were found.
  uint32_t GetNameIndexes(ConstString symbol_name,
                          std::vector<uint32_t> &indexes);

  void SymbolIndicesToSymbolContextList(std::vector<uint32_t> &symbol_indexes,
                                        SymbolContextList &sc_list);

  void RegisterMangledNameEntry(
      uint32_t value, std::set<const char *> &class_contexts,
      std::vector<std::pair<NameToIndexMap::Entry, const char *>> &backlog,
      RichManglingContext &rmc);

  void RegisterBacklogEntry(const NameToIndexMap::Entry &entry,
                            const char *decl_context,
                            const std::set<const char *> &class_contexts);

  Symtab(const Symtab &) = delete;
  const Symtab &operator=(const Symtab &) = delete;
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_SYMTAB_H
