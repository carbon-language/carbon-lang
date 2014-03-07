//===-- llvm/CodeGen/DIEHash.h - Dwarf Hashing Framework -------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for DWARF4 hashing of DIEs.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DIEHASH_H__
#define CODEGEN_ASMPRINTER_DIEHASH_H__

#include "DIE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MD5.h"

namespace llvm {

class AsmPrinter;
class CompileUnit;

/// \brief An object containing the capability of hashing and adding hash
/// attributes onto a DIE.
class DIEHash {

  // The entry for a particular attribute.
  struct AttrEntry {
    const DIEValue *Val;
    const DIEAbbrevData *Desc;
  };

  // Collection of all attributes used in hashing a particular DIE.
  struct DIEAttrs {
    AttrEntry DW_AT_name;
    AttrEntry DW_AT_accessibility;
    AttrEntry DW_AT_address_class;
    AttrEntry DW_AT_allocated;
    AttrEntry DW_AT_artificial;
    AttrEntry DW_AT_associated;
    AttrEntry DW_AT_binary_scale;
    AttrEntry DW_AT_bit_offset;
    AttrEntry DW_AT_bit_size;
    AttrEntry DW_AT_bit_stride;
    AttrEntry DW_AT_byte_size;
    AttrEntry DW_AT_byte_stride;
    AttrEntry DW_AT_const_expr;
    AttrEntry DW_AT_const_value;
    AttrEntry DW_AT_containing_type;
    AttrEntry DW_AT_count;
    AttrEntry DW_AT_data_bit_offset;
    AttrEntry DW_AT_data_location;
    AttrEntry DW_AT_data_member_location;
    AttrEntry DW_AT_decimal_scale;
    AttrEntry DW_AT_decimal_sign;
    AttrEntry DW_AT_default_value;
    AttrEntry DW_AT_digit_count;
    AttrEntry DW_AT_discr;
    AttrEntry DW_AT_discr_list;
    AttrEntry DW_AT_discr_value;
    AttrEntry DW_AT_encoding;
    AttrEntry DW_AT_enum_class;
    AttrEntry DW_AT_endianity;
    AttrEntry DW_AT_explicit;
    AttrEntry DW_AT_is_optional;
    AttrEntry DW_AT_location;
    AttrEntry DW_AT_lower_bound;
    AttrEntry DW_AT_mutable;
    AttrEntry DW_AT_ordering;
    AttrEntry DW_AT_picture_string;
    AttrEntry DW_AT_prototyped;
    AttrEntry DW_AT_small;
    AttrEntry DW_AT_segment;
    AttrEntry DW_AT_string_length;
    AttrEntry DW_AT_threads_scaled;
    AttrEntry DW_AT_upper_bound;
    AttrEntry DW_AT_use_location;
    AttrEntry DW_AT_use_UTF8;
    AttrEntry DW_AT_variable_parameter;
    AttrEntry DW_AT_virtuality;
    AttrEntry DW_AT_visibility;
    AttrEntry DW_AT_vtable_elem_location;
    AttrEntry DW_AT_type;

    // Insert any additional ones here...
  };

public:
  DIEHash(AsmPrinter *A = NULL) : AP(A) {}

  /// \brief Computes the ODR signature.
  uint64_t computeDIEODRSignature(const DIE &Die);

  /// \brief Computes the CU signature.
  uint64_t computeCUSignature(const DIE &Die);

  /// \brief Computes the type signature.
  uint64_t computeTypeSignature(const DIE &Die);

  // Helper routines to process parts of a DIE.
private:
  /// \brief Adds the parent context of \param Die to the hash.
  void addParentContext(const DIE &Die);

  /// \brief Adds the attributes of \param Die to the hash.
  void addAttributes(const DIE &Die);

  /// \brief Computes the full DWARF4 7.27 hash of the DIE.
  void computeHash(const DIE &Die);

  // Routines that add DIEValues to the hash.
public:
  /// \brief Adds \param Value to the hash.
  void update(uint8_t Value) { Hash.update(Value); }

  /// \brief Encodes and adds \param Value to the hash as a ULEB128.
  void addULEB128(uint64_t Value);

  /// \brief Encodes and adds \param Value to the hash as a SLEB128.
  void addSLEB128(int64_t Value);

private:
  /// \brief Adds \param Str to the hash and includes a NULL byte.
  void addString(StringRef Str);

  /// \brief Collects the attributes of DIE \param Die into the \param Attrs
  /// structure.
  void collectAttributes(const DIE &Die, DIEAttrs &Attrs);

  /// \brief Hashes the attributes in \param Attrs in order.
  void hashAttributes(const DIEAttrs &Attrs, dwarf::Tag Tag);

  /// \brief Hashes the data in a block like DIEValue, e.g. DW_FORM_block or
  /// DW_FORM_exprloc.
  void hashBlockData(const SmallVectorImpl<DIEValue *> &Values);

  /// \brief Hashes an individual attribute.
  void hashAttribute(AttrEntry Attr, dwarf::Tag Tag);

  /// \brief Hashes an attribute that refers to another DIE.
  void hashDIEEntry(dwarf::Attribute Attribute, dwarf::Tag Tag,
                    const DIE &Entry);

  /// \brief Hashes a reference to a named type in such a way that is
  /// independent of whether that type is described by a declaration or a
  /// definition.
  void hashShallowTypeReference(dwarf::Attribute Attribute, const DIE &Entry,
                                StringRef Name);

  /// \brief Hashes a reference to a previously referenced type DIE.
  void hashRepeatedTypeReference(dwarf::Attribute Attribute,
                                 unsigned DieNumber);

  void hashNestedType(const DIE &Die, StringRef Name);

private:
  MD5 Hash;
  AsmPrinter *AP;
  DenseMap<const DIE *, unsigned> Numbering;
};
}

#endif
