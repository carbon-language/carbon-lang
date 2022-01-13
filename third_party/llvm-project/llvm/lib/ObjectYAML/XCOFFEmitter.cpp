//===- yaml2xcoff - Convert YAML to a xcoff object file -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The xcoff component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;

namespace {

constexpr unsigned DefaultSectionAlign = 4;
constexpr int16_t MaxSectionIndex = INT16_MAX;
constexpr uint32_t MaxRawDataSize = UINT32_MAX;

class XCOFFWriter {
public:
  XCOFFWriter(XCOFFYAML::Object &Obj, raw_ostream &OS, yaml::ErrorHandler EH)
      : Obj(Obj), W(OS, support::big), ErrHandler(EH),
        Strings(StringTableBuilder::XCOFF) {
    Is64Bit = Obj.Header.Magic == (llvm::yaml::Hex16)XCOFF::XCOFF64;
  }
  bool writeXCOFF();

private:
  bool nameShouldBeInStringTable(StringRef SymbolName);
  bool initFileHeader(uint64_t CurrentOffset);
  bool initSectionHeader(uint64_t &CurrentOffset);
  bool initRelocations(uint64_t &CurrentOffset);
  bool assignAddressesAndIndices();
  void writeFileHeader();
  void writeSectionHeader();
  bool writeSectionData();
  bool writeRelocations();
  bool writeSymbols();

  XCOFFYAML::Object &Obj;
  bool Is64Bit = false;
  support::endian::Writer W;
  yaml::ErrorHandler ErrHandler;
  StringTableBuilder Strings;
  uint64_t StartOffset;
  // Map the section name to its corrresponding section index.
  DenseMap<StringRef, int16_t> SectionIndexMap = {
      {StringRef("N_DEBUG"), XCOFF::N_DEBUG},
      {StringRef("N_ABS"), XCOFF::N_ABS},
      {StringRef("N_UNDEF"), XCOFF::N_UNDEF}};
  XCOFFYAML::FileHeader InitFileHdr = Obj.Header;
  std::vector<XCOFFYAML::Section> InitSections = Obj.Sections;
};

static void writeName(StringRef StrName, support::endian::Writer W) {
  char Name[XCOFF::NameSize];
  memset(Name, 0, XCOFF::NameSize);
  char SrcName[] = "";
  memcpy(Name, StrName.size() ? StrName.data() : SrcName, StrName.size());
  ArrayRef<char> NameRef(Name, XCOFF::NameSize);
  W.write(NameRef);
}

bool XCOFFWriter::nameShouldBeInStringTable(StringRef SymbolName) {
  // For XCOFF64: The symbol name is always in the string table.
  return (SymbolName.size() > XCOFF::NameSize) || Is64Bit;
}

bool XCOFFWriter::initRelocations(uint64_t &CurrentOffset) {
  for (uint16_t I = 0, E = InitSections.size(); I < E; ++I) {
    if (!InitSections[I].Relocations.empty()) {
      InitSections[I].NumberOfRelocations = InitSections[I].Relocations.size();
      InitSections[I].FileOffsetToRelocations = CurrentOffset;
      uint64_t RelSize = Is64Bit ? XCOFF::RelocationSerializationSize64
                                 : XCOFF::RelocationSerializationSize32;
      CurrentOffset += InitSections[I].NumberOfRelocations * RelSize;
      if (CurrentOffset > MaxRawDataSize) {
        ErrHandler("maximum object size of" + Twine(MaxRawDataSize) +
                   "exceeded when writing relocation data");
        return false;
      }
    }
  }
  return true;
}

bool XCOFFWriter::initSectionHeader(uint64_t &CurrentOffset) {
  uint64_t CurrentSecAddr = 0;
  for (uint16_t I = 0, E = InitSections.size(); I < E; ++I) {
    if (CurrentOffset > MaxRawDataSize) {
      ErrHandler("maximum object size of" + Twine(MaxRawDataSize) +
                 "exceeded when writing section data");
      return false;
    }

    // Assign indices for sections.
    if (InitSections[I].SectionName.size() &&
        !SectionIndexMap[InitSections[I].SectionName]) {
      // The section index starts from 1.
      SectionIndexMap[InitSections[I].SectionName] = I + 1;
      if ((I + 1) > MaxSectionIndex) {
        ErrHandler("exceeded the maximum permitted section index of " +
                   Twine(MaxSectionIndex));
        return false;
      }
    }

    // Calculate the physical/virtual address. This field should contain 0 for
    // all sections except the text, data and bss sections.
    if (InitSections[I].Flags != XCOFF::STYP_TEXT &&
        InitSections[I].Flags != XCOFF::STYP_DATA &&
        InitSections[I].Flags != XCOFF::STYP_BSS)
      InitSections[I].Address = 0;
    else
      InitSections[I].Address = CurrentSecAddr;

    // Calculate the FileOffsetToData and data size for sections.
    if (InitSections[I].SectionData.binary_size()) {
      InitSections[I].FileOffsetToData = CurrentOffset;
      CurrentOffset += InitSections[I].SectionData.binary_size();
      // Ensure the offset is aligned to DefaultSectionAlign.
      CurrentOffset = alignTo(CurrentOffset, DefaultSectionAlign);
      InitSections[I].Size = CurrentOffset - InitSections[I].FileOffsetToData;
      CurrentSecAddr += InitSections[I].Size;
    }
  }
  return initRelocations(CurrentOffset);
}

bool XCOFFWriter::initFileHeader(uint64_t CurrentOffset) {
  // The default format of the object file is XCOFF32.
  InitFileHdr.Magic = XCOFF::XCOFF32;
  InitFileHdr.NumberOfSections = Obj.Sections.size();
  InitFileHdr.NumberOfSymTableEntries = Obj.Symbols.size();

  for (const XCOFFYAML::Symbol &YamlSym : Obj.Symbols) {
    // Add the number of auxiliary symbols to the total number.
    InitFileHdr.NumberOfSymTableEntries += YamlSym.NumberOfAuxEntries;
    if (nameShouldBeInStringTable(YamlSym.SymbolName))
      Strings.add(YamlSym.SymbolName);
  }
  // Finalize the string table.
  Strings.finalize();

  // Calculate SymbolTableOffset for the file header.
  if (InitFileHdr.NumberOfSymTableEntries) {
    InitFileHdr.SymbolTableOffset = CurrentOffset;
    CurrentOffset +=
        InitFileHdr.NumberOfSymTableEntries * XCOFF::SymbolTableEntrySize;
    if (CurrentOffset > MaxRawDataSize) {
      ErrHandler("maximum object size of" + Twine(MaxRawDataSize) +
                 "exceeded when writing symbols");
      return false;
    }
  }
  // TODO: Calculate FileOffsetToLineNumbers when line number supported.
  return true;
}

bool XCOFFWriter::assignAddressesAndIndices() {
  Strings.clear();
  uint64_t FileHdrSize =
      Is64Bit ? XCOFF::FileHeaderSize64 : XCOFF::FileHeaderSize32;
  uint64_t SecHdrSize =
      Is64Bit ? XCOFF::SectionHeaderSize64 : XCOFF::SectionHeaderSize32;
  uint64_t CurrentOffset = FileHdrSize /* TODO: + auxiliaryHeaderSize() */ +
                           InitSections.size() * SecHdrSize;

  // Calculate section header info.
  if (!initSectionHeader(CurrentOffset))
    return false;
  // Calculate file header info.
  return initFileHeader(CurrentOffset);
}

void XCOFFWriter::writeFileHeader() {
  W.write<uint16_t>(Obj.Header.Magic ? Obj.Header.Magic : InitFileHdr.Magic);
  W.write<uint16_t>(Obj.Header.NumberOfSections ? Obj.Header.NumberOfSections
                                                : InitFileHdr.NumberOfSections);
  W.write<int32_t>(Obj.Header.TimeStamp);
  if (Is64Bit) {
    W.write<uint64_t>(Obj.Header.SymbolTableOffset
                          ? Obj.Header.SymbolTableOffset
                          : InitFileHdr.SymbolTableOffset);
    W.write<uint16_t>(Obj.Header.AuxHeaderSize);
    W.write<uint16_t>(Obj.Header.Flags);
    W.write<int32_t>(Obj.Header.NumberOfSymTableEntries
                         ? Obj.Header.NumberOfSymTableEntries
                         : InitFileHdr.NumberOfSymTableEntries);
  } else {
    W.write<uint32_t>(Obj.Header.SymbolTableOffset
                          ? Obj.Header.SymbolTableOffset
                          : InitFileHdr.SymbolTableOffset);
    W.write<int32_t>(Obj.Header.NumberOfSymTableEntries
                         ? Obj.Header.NumberOfSymTableEntries
                         : InitFileHdr.NumberOfSymTableEntries);
    W.write<uint16_t>(Obj.Header.AuxHeaderSize);
    W.write<uint16_t>(Obj.Header.Flags);
  }
}

void XCOFFWriter::writeSectionHeader() {
  for (uint16_t I = 0, E = Obj.Sections.size(); I < E; ++I) {
    XCOFFYAML::Section YamlSec = Obj.Sections[I];
    XCOFFYAML::Section DerivedSec = InitSections[I];
    writeName(YamlSec.SectionName, W);
    // Virtual address is the same as physical address.
    uint64_t SectionAddress =
        YamlSec.Address ? YamlSec.Address : DerivedSec.Address;
    if (Is64Bit) {
      W.write<uint64_t>(SectionAddress); // Physical address
      W.write<uint64_t>(SectionAddress); // Virtual address
      W.write<uint64_t>(YamlSec.Size ? YamlSec.Size : DerivedSec.Size);
      W.write<uint64_t>(YamlSec.FileOffsetToData ? YamlSec.FileOffsetToData
                                                 : DerivedSec.FileOffsetToData);
      W.write<uint64_t>(YamlSec.FileOffsetToRelocations
                            ? YamlSec.FileOffsetToRelocations
                            : DerivedSec.FileOffsetToRelocations);
      W.write<uint64_t>(YamlSec.FileOffsetToLineNumbers);
      W.write<uint32_t>(YamlSec.NumberOfRelocations
                            ? YamlSec.NumberOfRelocations
                            : DerivedSec.NumberOfRelocations);
      W.write<uint32_t>(YamlSec.NumberOfLineNumbers);
      W.write<int32_t>(YamlSec.Flags);
      W.OS.write_zeros(4);
    } else {
      W.write<uint32_t>(SectionAddress); // Physical address
      W.write<uint32_t>(SectionAddress); // Virtual address
      W.write<uint32_t>(YamlSec.Size ? YamlSec.Size : DerivedSec.Size);
      W.write<uint32_t>(YamlSec.FileOffsetToData ? YamlSec.FileOffsetToData
                                                 : DerivedSec.FileOffsetToData);
      W.write<uint32_t>(YamlSec.FileOffsetToRelocations
                            ? YamlSec.FileOffsetToRelocations
                            : DerivedSec.FileOffsetToRelocations);
      W.write<uint32_t>(YamlSec.FileOffsetToLineNumbers);
      W.write<uint16_t>(YamlSec.NumberOfRelocations
                            ? YamlSec.NumberOfRelocations
                            : DerivedSec.NumberOfRelocations);
      W.write<uint16_t>(YamlSec.NumberOfLineNumbers);
      W.write<int32_t>(YamlSec.Flags);
    }
  }
}

bool XCOFFWriter::writeSectionData() {
  for (uint16_t I = 0, E = Obj.Sections.size(); I < E; ++I) {
    XCOFFYAML::Section YamlSec = Obj.Sections[I];
    if (YamlSec.SectionData.binary_size()) {
      // Fill the padding size with zeros.
      int64_t PaddingSize =
          InitSections[I].FileOffsetToData - (W.OS.tell() - StartOffset);
      if (PaddingSize < 0) {
        ErrHandler("redundant data was written before section data");
        return false;
      }
      W.OS.write_zeros(PaddingSize);
      YamlSec.SectionData.writeAsBinary(W.OS);
    }
  }
  return true;
}

bool XCOFFWriter::writeRelocations() {
  for (uint16_t I = 0, E = Obj.Sections.size(); I < E; ++I) {
    XCOFFYAML::Section YamlSec = Obj.Sections[I];
    if (!YamlSec.Relocations.empty()) {
      int64_t PaddingSize =
          InitSections[I].FileOffsetToRelocations - (W.OS.tell() - StartOffset);
      if (PaddingSize < 0) {
        ErrHandler("redundant data was written before relocations");
        return false;
      }
      W.OS.write_zeros(PaddingSize);
      for (const XCOFFYAML::Relocation &YamlRel : YamlSec.Relocations) {
        if (Is64Bit)
          W.write<uint64_t>(YamlRel.VirtualAddress);
        else
          W.write<uint32_t>(YamlRel.VirtualAddress);
        W.write<uint32_t>(YamlRel.SymbolIndex);
        W.write<uint8_t>(YamlRel.Info);
        W.write<uint8_t>(YamlRel.Type);
      }
    }
  }
  return true;
}

bool XCOFFWriter::writeSymbols() {
  int64_t PaddingSize =
      (uint64_t)InitFileHdr.SymbolTableOffset - (W.OS.tell() - StartOffset);
  if (PaddingSize < 0) {
    ErrHandler("redundant data was written before symbols");
    return false;
  }
  W.OS.write_zeros(PaddingSize);
  for (const XCOFFYAML::Symbol &YamlSym : Obj.Symbols) {
    if (Is64Bit) {
      W.write<uint64_t>(YamlSym.Value);
      W.write<uint32_t>(Strings.getOffset(YamlSym.SymbolName));
    } else {
      if (nameShouldBeInStringTable(YamlSym.SymbolName)) {
        // For XCOFF32: A value of 0 indicates that the symbol name is in the
        // string table.
        W.write<int32_t>(0);
        W.write<uint32_t>(Strings.getOffset(YamlSym.SymbolName));
      } else {
        writeName(YamlSym.SymbolName, W);
      }
      W.write<uint32_t>(YamlSym.Value);
    }
    W.write<int16_t>(
        YamlSym.SectionName.size() ? SectionIndexMap[YamlSym.SectionName] : 0);
    W.write<uint16_t>(YamlSym.Type);
    W.write<uint8_t>(YamlSym.StorageClass);
    W.write<uint8_t>(YamlSym.NumberOfAuxEntries);

    // Now output the auxiliary entry.
    for (uint8_t I = 0, E = YamlSym.NumberOfAuxEntries; I < E; ++I) {
      // TODO: Auxiliary entry is not supported yet.
      // The auxiliary entries for a symbol follow its symbol table entry. The
      // length of each auxiliary entry is the same as a symbol table entry (18
      // bytes). The format and quantity of auxiliary entries depend on the
      // storage class (n_sclass) and type (n_type) of the symbol table entry.
      W.OS.write_zeros(XCOFF::SymbolTableEntrySize);
    }
  }
  return true;
}

bool XCOFFWriter::writeXCOFF() {
  if (!assignAddressesAndIndices())
    return false;
  StartOffset = W.OS.tell();
  writeFileHeader();
  if (!Obj.Sections.empty()) {
    writeSectionHeader();
    if (!writeSectionData())
      return false;
    if (!writeRelocations())
      return false;
  }
  if (!Obj.Symbols.empty() && !writeSymbols())
    return false;
  // Write the string table.
  if (Strings.getSize() > 4)
    Strings.write(W.OS);
  return true;
}

} // end anonymous namespace

namespace llvm {
namespace yaml {

bool yaml2xcoff(XCOFFYAML::Object &Doc, raw_ostream &Out, ErrorHandler EH) {
  XCOFFWriter Writer(Doc, Out, EH);
  return Writer.writeXCOFF();
}

} // namespace yaml
} // namespace llvm
