//===- yaml2dwarf - Convert YAML to DWARF binary data ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The DWARF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SwapByteOrder.h"

using namespace llvm;

template <typename T>
void writeInteger(T Integer, raw_ostream &OS, bool IsLittleEndian) {
  if (IsLittleEndian != sys::IsLittleEndianHost)
    sys::swapByteOrder(Integer);
  OS.write(reinterpret_cast<char *>(&Integer), sizeof(T));
}

void writeVariableSizedInteger(uint64_t Integer, size_t Size, raw_ostream &OS,
                               bool IsLittleEndian) {
  if (8 == Size)
    writeInteger((uint64_t)Integer, OS, IsLittleEndian);
  else if (4 == Size)
    writeInteger((uint32_t)Integer, OS, IsLittleEndian);
  else if (2 == Size)
    writeInteger((uint16_t)Integer, OS, IsLittleEndian);
  else if (1 == Size)
    writeInteger((uint8_t)Integer, OS, IsLittleEndian);
  else
    assert(false && "Invalid integer write size.");
}

void ZeroFillBytes(raw_ostream &OS, size_t Size) {
  std::vector<uint8_t> FillData;
  FillData.insert(FillData.begin(), Size, 0);
  OS.write(reinterpret_cast<char *>(FillData.data()), Size);
}

void yaml2debug_str(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Str : DI.DebugStrings) {
    OS.write(Str.data(), Str.size());
    OS.write('\0');
  }
}

void yaml2debug_abbrev(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto AbbrevDecl : DI.AbbrevDecls) {
    encodeULEB128(AbbrevDecl.Code, OS);
    encodeULEB128(AbbrevDecl.Tag, OS);
    OS.write(AbbrevDecl.Children);
    for (auto Attr : AbbrevDecl.Attributes) {
      encodeULEB128(Attr.Attribute, OS);
      encodeULEB128(Attr.Form, OS);
    }
    encodeULEB128(0, OS);
    encodeULEB128(0, OS);
  }
}

void yaml2debug_aranges(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Range : DI.ARanges) {
    auto HeaderStart = OS.tell();
    writeInteger((uint32_t)Range.Length, OS, DI.IsLittleEndian);
    writeInteger((uint16_t)Range.Version, OS, DI.IsLittleEndian);
    writeInteger((uint32_t)Range.CuOffset, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)Range.AddrSize, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)Range.SegSize, OS, DI.IsLittleEndian);

    auto HeaderSize = OS.tell() - HeaderStart;
    auto FirstDescriptor = alignTo(HeaderSize, Range.AddrSize * 2);
    ZeroFillBytes(OS, FirstDescriptor - HeaderSize);

    for (auto Descriptor : Range.Descriptors) {
      writeVariableSizedInteger(Descriptor.Address, Range.AddrSize, OS,
                                DI.IsLittleEndian);
      writeVariableSizedInteger(Descriptor.Length, Range.AddrSize, OS,
                                DI.IsLittleEndian);
    }
    ZeroFillBytes(OS, Range.AddrSize * 2);
  }
}

void yaml2pubsection(raw_ostream &OS, const DWARFYAML::PubSection &Sect,
                     bool IsLittleEndian) {
  writeInteger((uint32_t)Sect.Length, OS, IsLittleEndian);
  writeInteger((uint16_t)Sect.Version, OS, IsLittleEndian);
  writeInteger((uint32_t)Sect.UnitOffset, OS, IsLittleEndian);
  writeInteger((uint32_t)Sect.UnitSize, OS, IsLittleEndian);
  for (auto Entry : Sect.Entries) {
    writeInteger((uint32_t)Entry.DieOffset, OS, IsLittleEndian);
    if (Sect.IsGNUStyle)
      writeInteger((uint32_t)Entry.Descriptor, OS, IsLittleEndian);
    OS.write(Entry.Name.data(), Entry.Name.size());
    OS.write('\0');
  }
}