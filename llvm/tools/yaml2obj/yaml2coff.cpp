//===- yaml2coff - Convert YAML to a COFF object file ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The COFF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/COFFYAML.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

using namespace llvm;

/// This parses a yaml stream that represents a COFF object file.
/// See docs/yaml2obj for the yaml scheema.
struct COFFParser {
  COFFParser(COFFYAML::Object &Obj) : Obj(Obj) {
    // A COFF string table always starts with a 4 byte size field. Offsets into
    // it include this size, so allocate it now.
    StringTable.append(4, 0);
  }

  bool parseSections() {
    for (std::vector<COFFYAML::Section>::iterator i = Obj.Sections.begin(),
           e = Obj.Sections.end(); i != e; ++i) {
      COFFYAML::Section &Sec = *i;

      // If the name is less than 8 bytes, store it in place, otherwise
      // store it in the string table.
      StringRef Name = Sec.Name;

      if (Name.size() <= COFF::NameSize) {
        std::copy(Name.begin(), Name.end(), Sec.Header.Name);
      } else {
        // Add string to the string table and format the index for output.
        unsigned Index = getStringIndex(Name);
        std::string str = utostr(Index);
        if (str.size() > 7) {
          errs() << "String table got too large";
          return false;
        }
        Sec.Header.Name[0] = '/';
        std::copy(str.begin(), str.end(), Sec.Header.Name + 1);
      }

      Sec.Header.Characteristics |= (Log2_32(Sec.Alignment) + 1) << 20;
    }
    return true;
  }

  bool parseSymbols() {
    for (std::vector<COFFYAML::Symbol>::iterator i = Obj.Symbols.begin(),
           e = Obj.Symbols.end(); i != e; ++i) {
      COFFYAML::Symbol &Sym = *i;

      // If the name is less than 8 bytes, store it in place, otherwise
      // store it in the string table.
      StringRef Name = Sym.Name;
      if (Name.size() <= COFF::NameSize) {
        std::copy(Name.begin(), Name.end(), Sym.Header.Name);
      } else {
        // Add string to the string table and format the index for output.
        unsigned Index = getStringIndex(Name);
        *reinterpret_cast<support::aligned_ulittle32_t*>(
            Sym.Header.Name + 4) = Index;
      }

      Sym.Header.Type = Sym.SimpleType;
      Sym.Header.Type |= Sym.ComplexType << COFF::SCT_COMPLEX_TYPE_SHIFT;
    }
    return true;
  }

  bool parse() {
    if (!parseSections())
      return false;
    if (!parseSymbols())
      return false;
    return true;
  }

  unsigned getStringIndex(StringRef Str) {
    StringMap<unsigned>::iterator i = StringTableMap.find(Str);
    if (i == StringTableMap.end()) {
      unsigned Index = StringTable.size();
      StringTable.append(Str.begin(), Str.end());
      StringTable.push_back(0);
      StringTableMap[Str] = Index;
      return Index;
    }
    return i->second;
  }

  COFFYAML::Object &Obj;

  StringMap<unsigned> StringTableMap;
  std::string StringTable;
};

// Take a CP and assign addresses and sizes to everything. Returns false if the
// layout is not valid to do.
static bool layoutCOFF(COFFParser &CP) {
  uint32_t SectionTableStart = 0;
  uint32_t SectionTableSize  = 0;

  // The section table starts immediately after the header, including the
  // optional header.
  SectionTableStart = sizeof(COFF::header) + CP.Obj.Header.SizeOfOptionalHeader;
  SectionTableSize = sizeof(COFF::section) * CP.Obj.Sections.size();

  uint32_t CurrentSectionDataOffset = SectionTableStart + SectionTableSize;

  // Assign each section data address consecutively.
  for (std::vector<COFFYAML::Section>::iterator i = CP.Obj.Sections.begin(),
                                                e = CP.Obj.Sections.end();
                                                i != e; ++i) {
    if (i->SectionData.binary_size() > 0) {
      i->Header.SizeOfRawData = i->SectionData.binary_size();
      i->Header.PointerToRawData = CurrentSectionDataOffset;
      CurrentSectionDataOffset += i->Header.SizeOfRawData;
      if (!i->Relocations.empty()) {
        i->Header.PointerToRelocations = CurrentSectionDataOffset;
        i->Header.NumberOfRelocations = i->Relocations.size();
        CurrentSectionDataOffset += i->Header.NumberOfRelocations *
          COFF::RelocationSize;
      }
      // TODO: Handle alignment.
    } else {
      i->Header.SizeOfRawData = 0;
      i->Header.PointerToRawData = 0;
    }
  }

  uint32_t SymbolTableStart = CurrentSectionDataOffset;

  // Calculate number of symbols.
  uint32_t NumberOfSymbols = 0;
  for (std::vector<COFFYAML::Symbol>::iterator i = CP.Obj.Symbols.begin(),
                                               e = CP.Obj.Symbols.end();
                                               i != e; ++i) {
    unsigned AuxBytes = i->AuxiliaryData.binary_size();
    if (AuxBytes % COFF::SymbolSize != 0) {
      errs() << "AuxiliaryData size not a multiple of symbol size!\n";
      return false;
    }
    i->Header.NumberOfAuxSymbols = AuxBytes / COFF::SymbolSize;
    NumberOfSymbols += 1 + i->Header.NumberOfAuxSymbols;
  }

  // Store all the allocated start addresses in the header.
  CP.Obj.Header.NumberOfSections = CP.Obj.Sections.size();
  CP.Obj.Header.NumberOfSymbols = NumberOfSymbols;
  CP.Obj.Header.PointerToSymbolTable = SymbolTableStart;

  *reinterpret_cast<support::ulittle32_t *>(&CP.StringTable[0])
    = CP.StringTable.size();

  return true;
}

template <typename value_type>
struct binary_le_impl {
  value_type Value;
  binary_le_impl(value_type V) : Value(V) {}
};

template <typename value_type>
raw_ostream &operator <<( raw_ostream &OS
                        , const binary_le_impl<value_type> &BLE) {
  char Buffer[sizeof(BLE.Value)];
  support::endian::write<value_type, support::little, support::unaligned>(
    Buffer, BLE.Value);
  OS.write(Buffer, sizeof(BLE.Value));
  return OS;
}

template <typename value_type>
binary_le_impl<value_type> binary_le(value_type V) {
  return binary_le_impl<value_type>(V);
}

bool writeCOFF(COFFParser &CP, raw_ostream &OS) {
  OS << binary_le(CP.Obj.Header.Machine)
     << binary_le(CP.Obj.Header.NumberOfSections)
     << binary_le(CP.Obj.Header.TimeDateStamp)
     << binary_le(CP.Obj.Header.PointerToSymbolTable)
     << binary_le(CP.Obj.Header.NumberOfSymbols)
     << binary_le(CP.Obj.Header.SizeOfOptionalHeader)
     << binary_le(CP.Obj.Header.Characteristics);

  // Output section table.
  for (std::vector<COFFYAML::Section>::iterator i = CP.Obj.Sections.begin(),
                                                e = CP.Obj.Sections.end();
                                                i != e; ++i) {
    OS.write(i->Header.Name, COFF::NameSize);
    OS << binary_le(i->Header.VirtualSize)
       << binary_le(i->Header.VirtualAddress)
       << binary_le(i->Header.SizeOfRawData)
       << binary_le(i->Header.PointerToRawData)
       << binary_le(i->Header.PointerToRelocations)
       << binary_le(i->Header.PointerToLineNumbers)
       << binary_le(i->Header.NumberOfRelocations)
       << binary_le(i->Header.NumberOfLineNumbers)
       << binary_le(i->Header.Characteristics);
  }

  unsigned CurSymbol = 0;
  StringMap<unsigned> SymbolTableIndexMap;
  for (std::vector<COFFYAML::Symbol>::iterator I = CP.Obj.Symbols.begin(),
                                               E = CP.Obj.Symbols.end();
       I != E; ++I) {
    SymbolTableIndexMap[I->Name] = CurSymbol;
    CurSymbol += 1 + I->Header.NumberOfAuxSymbols;
  }

  // Output section data.
  for (std::vector<COFFYAML::Section>::iterator i = CP.Obj.Sections.begin(),
                                                e = CP.Obj.Sections.end();
                                                i != e; ++i) {
    i->SectionData.writeAsBinary(OS);
    for (unsigned I2 = 0, E2 = i->Relocations.size(); I2 != E2; ++I2) {
      const COFFYAML::Relocation &R = i->Relocations[I2];
      uint32_t SymbolTableIndex = SymbolTableIndexMap[R.SymbolName];
      OS << binary_le(R.VirtualAddress)
         << binary_le(SymbolTableIndex)
         << binary_le(R.Type);
    }
  }

  // Output symbol table.

  for (std::vector<COFFYAML::Symbol>::const_iterator i = CP.Obj.Symbols.begin(),
                                                     e = CP.Obj.Symbols.end();
                                                     i != e; ++i) {
    OS.write(i->Header.Name, COFF::NameSize);
    OS << binary_le(i->Header.Value)
       << binary_le(i->Header.SectionNumber)
       << binary_le(i->Header.Type)
       << binary_le(i->Header.StorageClass)
       << binary_le(i->Header.NumberOfAuxSymbols);
    i->AuxiliaryData.writeAsBinary(OS);
  }

  // Output string table.
  OS.write(&CP.StringTable[0], CP.StringTable.size());
  return true;
}

int yaml2coff(llvm::raw_ostream &Out, llvm::MemoryBuffer *Buf) {
  yaml::Input YIn(Buf->getBuffer());
  COFFYAML::Object Doc;
  YIn >> Doc;
  if (YIn.error()) {
    errs() << "yaml2obj: Failed to parse YAML file!\n";
    return 1;
  }

  COFFParser CP(Doc);
  if (!CP.parse()) {
    errs() << "yaml2obj: Failed to parse YAML file!\n";
    return 1;
  }

  if (!layoutCOFF(CP)) {
    errs() << "yaml2obj: Failed to layout COFF file!\n";
    return 1;
  }
  if (!writeCOFF(CP, Out)) {
    errs() << "yaml2obj: Failed to write COFF file!\n";
    return 1;
  }
  return 0;
}
