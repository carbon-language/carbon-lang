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

#include "llvm/Support/MD5.h"

namespace llvm {

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
    AttrEntry DW_AT_language;
  };

public:
  /// \brief Computes the ODR signature
  uint64_t computeDIEODRSignature(DIE *Die);

  /// \brief Computes the CU signature
  uint64_t computeCUSignature(DIE *Die);

  // Helper routines to process parts of a DIE.
private:
  /// \brief Adds the parent context of \param Die to the hash.
  void addParentContext(DIE *Die);

  /// \brief Adds the attributes of \param Die to the hash.
  void addAttributes(DIE *Die);

  /// \brief Computes the full DWARF4 7.27 hash of the DIE.
  void computeHash(DIE *Die);

  // Routines that add DIEValues to the hash.
private:
  /// \brief Encodes and adds \param Value to the hash as a ULEB128.
  void addULEB128(uint64_t Value);

  /// \brief Adds \param Str to the hash and includes a NULL byte.
  void addString(StringRef Str);

  /// \brief Collects the attributes of DIE \param Die into the \param Attrs
  /// structure.
  void collectAttributes(DIE *Die, DIEAttrs &Attrs);

  /// \brief Hashes the attributes in \param Attrs in order.
  void hashAttributes(const DIEAttrs &Attrs);

  /// \brief Hashes an individual attribute.
  void hashAttribute(AttrEntry Attr);

private:
  MD5 Hash;
};
}
