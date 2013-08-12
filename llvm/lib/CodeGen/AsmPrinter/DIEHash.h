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
public:
  /// \brief Computes the ODR signature
  uint64_t computeDIEODRSignature(DIE *Die);

  // Helper routines to process parts of a DIE.
 private:
  /// \brief Adds the parent context of \param Die to the hash.
  void addParentContext(DIE *Die);
  
  // Routines that add DIEValues to the hash.
private:
  /// \brief Encodes and adds \param Value to the hash as a ULEB128.
  void addULEB128(uint64_t Value);

  /// \brief Adds \param Str to the hash and includes a NULL byte.
  void addString(StringRef Str);
  
private:
  MD5 Hash;
};
}
