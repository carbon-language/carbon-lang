//===-- DWARFDIE.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDIE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDIE_H

#include "DWARFBaseDIE.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/iterator_range.h"

class DWARFDIE : public DWARFBaseDIE {
public:
  class child_iterator;
  using DWARFBaseDIE::DWARFBaseDIE;

  // Tests
  bool IsStructUnionOrClass() const;

  bool IsMethod() const;

  // Accessors

  // Accessing information about a DIE
  const char *GetMangledName() const;

  const char *GetPubname() const;

  const char *GetQualifiedName(std::string &storage) const;

  using DWARFBaseDIE::GetName;
  void GetName(lldb_private::Stream &s) const;

  void AppendTypeName(lldb_private::Stream &s) const;

  lldb_private::Type *ResolveType() const;

  // Resolve a type by UID using this DIE's DWARF file
  lldb_private::Type *ResolveTypeUID(const DWARFDIE &die) const;

  // Functions for obtaining DIE relations and references

  DWARFDIE
  GetParent() const;

  DWARFDIE
  GetFirstChild() const;

  DWARFDIE
  GetSibling() const;

  DWARFDIE
  GetReferencedDIE(const dw_attr_t attr) const;

  // Get a another DIE from the same DWARF file as this DIE. This will
  // check the current DIE's compile unit first to see if "die_offset" is
  // in the same compile unit, and fall back to checking the DWARF file.
  DWARFDIE
  GetDIE(dw_offset_t die_offset) const;
  using DWARFBaseDIE::GetDIE;

  DWARFDIE
  LookupDeepestBlock(lldb::addr_t file_addr) const;

  DWARFDIE
  GetParentDeclContextDIE() const;

  // DeclContext related functions
  std::vector<DWARFDIE> GetDeclContextDIEs() const;

  /// Return this DIE's decl context as it is needed to look up types
  /// in Clang's -gmodules debug info format.
  void GetDeclContext(
      llvm::SmallVectorImpl<lldb_private::CompilerContext> &context) const;

  // Getting attribute values from the DIE.
  //
  // GetAttributeValueAsXXX() functions should only be used if you are
  // looking for one or two attributes on a DIE. If you are trying to
  // parse all attributes, use GetAttributes (...) instead
  DWARFDIE
  GetAttributeValueAsReferenceDIE(const dw_attr_t attr) const;

  bool GetDIENamesAndRanges(const char *&name, const char *&mangled,
                            DWARFRangeList &ranges, int &decl_file,
                            int &decl_line, int &decl_column, int &call_file,
                            int &call_line, int &call_column,
                            lldb_private::DWARFExpression *frame_base) const;

  /// The range of all the children of this DIE.
  llvm::iterator_range<child_iterator> children() const;
};

class DWARFDIE::child_iterator
    : public llvm::iterator_facade_base<DWARFDIE::child_iterator,
                                        std::forward_iterator_tag, DWARFDIE> {
  /// The current child or an invalid DWARFDie.
  DWARFDIE m_die;

public:
  child_iterator() = default;
  child_iterator(const DWARFDIE &parent) : m_die(parent.GetFirstChild()) {}
  bool operator==(const child_iterator &it) const {
    // DWARFDIE's operator== differentiates between an invalid DWARFDIE that
    // has a CU but no DIE and one that has neither CU nor DIE. The 'end'
    // iterator could be default constructed, so explicitly allow
    // (CU, (DIE)nullptr) == (nullptr, nullptr) -> true
    if (!m_die.IsValid() && !it.m_die.IsValid())
      return true;
    return m_die == it.m_die;
  }
  const DWARFDIE &operator*() const {
    assert(m_die.IsValid() && "Derefencing invalid iterator?");
    return m_die;
  }
  DWARFDIE &operator*() {
    assert(m_die.IsValid() && "Derefencing invalid iterator?");
    return m_die;
  }
  child_iterator &operator++() {
    assert(m_die.IsValid() && "Incrementing invalid iterator?");
    m_die = m_die.GetSibling();
    return *this;
  }
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFDIE_H
