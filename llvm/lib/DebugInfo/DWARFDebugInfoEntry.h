//===-- DWARFDebugInfoEntry.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGINFOENTRY_H
#define LLVM_DEBUGINFO_DWARFDEBUGINFOENTRY_H

#include "DWARFAbbreviationDeclaration.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class DWARFCompileUnit;
class DWARFContext;
class DWARFFormValue;

/// DWARFDebugInfoEntryMinimal - A DIE with only the minimum required data.
class DWARFDebugInfoEntryMinimal {
  /// Offset within the .debug_info of the start of this entry.
  uint64_t Offset;

  /// How many to subtract from "this" to get the parent.
  /// If zero this die has no parent.
  uint32_t ParentIdx;

  /// How many to add to "this" to get the sibling.
  uint32_t SiblingIdx;

  const DWARFAbbreviationDeclaration *AbbrevDecl;
public:
  void dump(raw_ostream &OS, const DWARFCompileUnit *cu,
            unsigned recurseDepth, unsigned indent = 0) const;
  void dumpAttribute(raw_ostream &OS, const DWARFCompileUnit *cu,
                     uint32_t *offset_ptr, uint16_t attr, uint16_t form,
                     unsigned indent = 0) const;

  bool extractFast(const DWARFCompileUnit *cu, const uint8_t *fixed_form_sizes,
                   uint32_t *offset_ptr);

  /// Extract a debug info entry for a given compile unit from the
  /// .debug_info and .debug_abbrev data starting at the given offset.
  bool extract(const DWARFCompileUnit *cu, uint32_t *offset_ptr);

  uint32_t getTag() const { return AbbrevDecl ? AbbrevDecl->getTag() : 0; }
  bool isNULL() const { return AbbrevDecl == 0; }
  uint64_t getOffset() const { return Offset; }
  uint32_t getNumAttributes() const {
    return !isNULL() ? AbbrevDecl->getNumAttributes() : 0;
  }
  bool hasChildren() const { return !isNULL() && AbbrevDecl->hasChildren(); }

  // We know we are kept in a vector of contiguous entries, so we know
  // our parent will be some index behind "this".
  DWARFDebugInfoEntryMinimal *getParent() {
    return ParentIdx > 0 ? this - ParentIdx : 0;
  }
  const DWARFDebugInfoEntryMinimal *getParent() const {
    return ParentIdx > 0 ? this - ParentIdx : 0;
  }
  // We know we are kept in a vector of contiguous entries, so we know
  // our sibling will be some index after "this".
  DWARFDebugInfoEntryMinimal *getSibling() {
    return SiblingIdx > 0 ? this + SiblingIdx : 0;
  }
  const DWARFDebugInfoEntryMinimal *getSibling() const {
    return SiblingIdx > 0 ? this + SiblingIdx : 0;
  }
  // We know we are kept in a vector of contiguous entries, so we know
  // we don't need to store our child pointer, if we have a child it will
  // be the next entry in the list...
  DWARFDebugInfoEntryMinimal *getFirstChild() {
    return hasChildren() ? this + 1 : 0;
  }
  const DWARFDebugInfoEntryMinimal *getFirstChild() const {
    return hasChildren() ? this + 1 : 0;
  }

  void setParent(DWARFDebugInfoEntryMinimal *parent) {
    if (parent) {
      // We know we are kept in a vector of contiguous entries, so we know
      // our parent will be some index behind "this".
      ParentIdx = this - parent;
    } else
      ParentIdx = 0;
  }
  void setSibling(DWARFDebugInfoEntryMinimal *sibling) {
    if (sibling) {
      // We know we are kept in a vector of contiguous entries, so we know
      // our sibling will be some index after "this".
      SiblingIdx = sibling - this;
      sibling->setParent(getParent());
    } else
      SiblingIdx = 0;
  }

  const DWARFAbbreviationDeclaration *getAbbreviationDeclarationPtr() const {
    return AbbrevDecl;
  }

  uint32_t getAttributeValue(const DWARFCompileUnit *cu,
                             const uint16_t attr, DWARFFormValue &formValue,
                             uint32_t *end_attr_offset_ptr = 0) const;

  const char* getAttributeValueAsString(const DWARFCompileUnit* cu,
                                        const uint16_t attr,
                                        const char *fail_value) const;

  uint64_t getAttributeValueAsUnsigned(const DWARFCompileUnit *cu,
                                       const uint16_t attr,
                                       uint64_t fail_value) const;

  uint64_t getAttributeValueAsReference(const DWARFCompileUnit *cu,
                                        const uint16_t attr,
                                        uint64_t fail_value) const;

  int64_t getAttributeValueAsSigned(const DWARFCompileUnit* cu,
                                    const uint16_t attr,
                                    int64_t fail_value) const;
};

}

#endif
