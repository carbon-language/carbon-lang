//===--- BinaryPatcher.h  - Classes for modifying sections of the binary --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interfaces for applying small modifications to parts of a binary file. Some
// specializations facilitate the modification of specific ELF/DWARF sections.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_PATCHER_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_PATCHER_H

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include <string>
#include <utility>
#include <vector>

namespace llvm {
namespace bolt {

/// Abstract interface for classes that apply modifications to a binary string.
class BinaryPatcher {
public:
  virtual ~BinaryPatcher() {}
  /// Applies in-place modifications to the binary string \p BinaryContents .
  virtual void patchBinary(std::string &BinaryContents) = 0;
};

/// Applies simple modifications to a binary string, such as directly replacing
/// the contents of a certain portion with a string or an integer.
class SimpleBinaryPatcher : public BinaryPatcher {
private:
  std::vector<std::pair<uint32_t, std::string>> Patches;

  /// Adds a patch to replace the contents of \p ByteSize bytes with the integer
  /// \p NewValue encoded in little-endian, with the least-significant byte
  /// being written at the offset \p Offset .
  void addLEPatch(uint32_t Offset, uint64_t NewValue, size_t ByteSize);

public:
  ~SimpleBinaryPatcher() {}

  /// Adds a patch to replace the contents of the binary string starting at the
  /// specified \p Offset with the string \p NewValue.
  void addBinaryPatch(uint32_t Offset, const std::string &NewValue);

  /// Adds a patch to replace the contents of a single byte of the string, at
  /// the offset \p Offset, with the value \Value .
  void addBytePatch(uint32_t Offset, uint8_t Value);

  /// Adds a patch to put the integer \p NewValue encoded as a 64-bit
  /// little-endian value at offset \p Offset.
  void addLE64Patch(uint32_t Offset, uint64_t NewValue);

  /// Adds a patch to put the integer \p NewValue encoded as a 32-bit
  /// little-endian value at offset \p Offset.
  void addLE32Patch(uint32_t Offset, uint32_t NewValue);

  void patchBinary(std::string &BinaryContents) override;
};

/// Apply small modifications to the .debug_abbrev DWARF section.
class DebugAbbrevPatcher : public BinaryPatcher {
private:
  /// Patch of changing one attribute to another.
  struct AbbrevAttrPatch {
    uint32_t Code;    // Code of abbreviation to be modified.
    uint16_t Attr;    // ID of attribute to be replaced.
    uint8_t NewAttr;  // ID of the new attribute.
    uint8_t NewForm;  // Form of the new attribute.
  };

  std::map<const DWARFUnit *, std::vector<AbbrevAttrPatch>> Patches;

public:
  ~DebugAbbrevPatcher() { }
  /// Adds a patch to change an attribute of an abbreviation that belongs to
  /// \p Unit to another attribute.
  /// \p AbbrevCode code of the abbreviation to be modified.
  /// \p AttrTag ID of the attribute to be replaced.
  /// \p NewAttrTag ID of the new attribute.
  /// \p NewAttrForm Form of the new attribute.
  /// We only handle standard forms, that are encoded in a single byte.
  void addAttributePatch(const DWARFUnit *Unit,
                         uint32_t AbbrevCode,
                         uint16_t AttrTag,
                         uint8_t NewAttrTag,
                         uint8_t NewAttrForm);

  void patchBinary(std::string &Contents) override;
};

} // namespace llvm
} // namespace bolt

#endif
