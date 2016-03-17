//===--- BinaryPatcher.h  - Classes for modifying sections of the binary --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryPatcher.h"
#include <algorithm>
#include <cassert>

namespace llvm {
namespace bolt {

void SimpleBinaryPatcher::addBinaryPatch(uint32_t Offset,
                                         const std::string &NewValue) {
  Patches.emplace_back(std::make_pair(Offset, NewValue));
}

void SimpleBinaryPatcher::addBytePatch(uint32_t Offset, uint8_t Value) {
  Patches.emplace_back(std::make_pair(Offset, std::string(1, Value)));
}

void SimpleBinaryPatcher::addLEPatch(uint32_t Offset, uint64_t NewValue,
                                     size_t ByteSize) {
  std::string LE64(ByteSize, 0);
  for (size_t I = 0; I < ByteSize; ++I) {
    LE64[I] = NewValue & 0xff;
    NewValue >>= 8;
  }
  Patches.emplace_back(std::make_pair(Offset, LE64));
}

void SimpleBinaryPatcher::addLE64Patch(uint32_t Offset, uint64_t NewValue) {
  addLEPatch(Offset, NewValue, 8);
}

void SimpleBinaryPatcher::addLE32Patch(uint32_t Offset, uint32_t NewValue) {
  addLEPatch(Offset, NewValue, 4);
}

void SimpleBinaryPatcher::patchBinary(std::string &BinaryContents) {
  for (const auto &Patch : Patches) {
    uint32_t Offset = Patch.first;
    const std::string &ByteSequence = Patch.second;
    assert(Offset + ByteSequence.size() <= BinaryContents.size() &&
        "Applied patch runs over binary size.");
    for (uint64_t I = 0, Size = ByteSequence.size(); I < Size; ++I) {
      BinaryContents[Offset + I] = ByteSequence[I];
    }
  }
}

void DebugAbbrevPatcher::addAttributePatch(const DWARFUnit *Unit,
                                           uint32_t AbbrevCode,
                                           uint16_t AttrTag,
                                           uint8_t NewAttrTag,
                                           uint8_t NewAttrForm) {
  assert(Unit && "No compile unit specified.");
  Patches[Unit].push_back(
      AbbrevAttrPatch{AbbrevCode, AttrTag, NewAttrTag, NewAttrForm});
}

void DebugAbbrevPatcher::patchBinary(std::string &Contents) {
  SimpleBinaryPatcher Patcher;

  for (const auto &UnitPatchesPair : Patches) {
    const auto *Unit = UnitPatchesPair.first;
    const auto *UnitAbbreviations = Unit->getAbbreviations();
    assert(UnitAbbreviations &&
           "Compile unit doesn't have associated abbreviations.");
    const auto &UnitPatches = UnitPatchesPair.second;
    for (const auto &AttrPatch : UnitPatches) {
      const auto *AbbreviationDeclaration =
        UnitAbbreviations->getAbbreviationDeclaration(AttrPatch.Code);
      assert(AbbreviationDeclaration && "No abbreviation with given code.");
      const auto *Attribute = AbbreviationDeclaration->findAttribute(
          AttrPatch.Attr);

      if (!Attribute) {
        errs() << "Attribute " << AttrPatch.Attr << " does not occur in "
               << " abbrev " << AttrPatch.Code << " of CU " << Unit->getOffset()
               << " in decl@" << AbbreviationDeclaration
               << " and index = " << AbbreviationDeclaration->findAttributeIndex(AttrPatch.Attr)
               << "\n";
        errs() << "Look at the abbrev:\n";
        AbbreviationDeclaration->dump(errs());

        assert(Attribute && "Specified attribute doesn't occur in abbreviation.");
      }
      // Because we're only handling standard values (i.e. no DW_FORM_GNU_* or
      // DW_AT_APPLE_*), they are all small (< 128) and encoded in a single
      // byte in ULEB128, otherwise it'll be more tricky as we may need to
      // grow or shrink the section.
      Patcher.addBytePatch(Attribute->AttrOffset,
          AttrPatch.NewAttr);
      Patcher.addBytePatch(Attribute->FormOffset,
          AttrPatch.NewForm);
    }
  }
  Patcher.patchBinary(Contents);
}

}  // namespace llvm
}  // namespace bolt
