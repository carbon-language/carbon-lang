//===- yaml2obj - Convert YAML to a binary object file --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program takes a YAML description of an object file and outputs the
// binary equivalent.
//
// This is used for writing tests that require binary files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include <vector>

using namespace llvm;

static cl::opt<std::string>
  Input(cl::Positional, cl::desc("<input>"), cl::init("-"));

// The structure of the yaml files is not an exact 1:1 match to COFF. In order
// to use yaml::IO, we use these structures which are closer to the source.
namespace COFFYAML {
  struct Section {
    COFF::section Header;
    StringRef SectionData;
    std::vector<COFF::relocation> Relocations;
    StringRef Name;
    Section() {
      memset(&Header, 0, sizeof(COFF::section));
    }
  };

  struct Symbol {
    COFF::symbol Header;
    COFF::SymbolBaseType SimpleType;
    COFF::SymbolComplexType ComplexType;
    StringRef AuxiliaryData;
    StringRef Name;
    Symbol() {
      memset(&Header, 0, sizeof(COFF::symbol));
    }
  };

  struct Object {
    COFF::header Header;
    std::vector<Section> Sections;
    std::vector<Symbol> Symbols;
    Object() {
      memset(&Header, 0, sizeof(COFF::header));
    }
  };
}

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
    if (!i->SectionData.empty()) {
      i->Header.SizeOfRawData = i->SectionData.size()/2;
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
    unsigned AuxBytes = i->AuxiliaryData.size() / 2;
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

static bool writeHexData(StringRef Data, raw_ostream &OS) {
  unsigned Size = Data.size();
  if (Size % 2)
    return false;

  for (unsigned I = 0; I != Size; I += 2) {
    uint8_t Byte;
    if (Data.substr(I,  2).getAsInteger(16, Byte))
      return false;
    OS.write(Byte);
  }

  return true;
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

  // Output section data.
  for (std::vector<COFFYAML::Section>::iterator i = CP.Obj.Sections.begin(),
                                                e = CP.Obj.Sections.end();
                                                i != e; ++i) {
    if (!i->SectionData.empty()) {
      if (!writeHexData(i->SectionData, OS)) {
        errs() << "SectionData must be a collection of pairs of hex bytes";
        return false;
      }
    }
    for (unsigned I2 = 0, E2 = i->Relocations.size(); I2 != E2; ++I2) {
      const COFF::relocation &R = i->Relocations[I2];
      OS << binary_le(R.VirtualAddress)
         << binary_le(R.SymbolTableIndex)
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
    if (!i->AuxiliaryData.empty()) {
      if (!writeHexData(i->AuxiliaryData, OS)) {
        errs() << "AuxiliaryData must be a collection of pairs of hex bytes";
        return false;
      }
    }
  }

  // Output string table.
  OS.write(&CP.StringTable[0], CP.StringTable.size());
  return true;
}

LLVM_YAML_IS_SEQUENCE_VECTOR(COFF::relocation)
LLVM_YAML_IS_SEQUENCE_VECTOR(COFFYAML::Section)
LLVM_YAML_IS_SEQUENCE_VECTOR(COFFYAML::Symbol)

namespace llvm {

namespace COFF {
  Characteristics operator|(Characteristics a, Characteristics b) {
    uint32_t Ret = static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
    return static_cast<Characteristics>(Ret);
  }

  SectionCharacteristics
  operator|(SectionCharacteristics a, SectionCharacteristics b) {
    uint32_t Ret = static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
    return static_cast<SectionCharacteristics>(Ret);
  }
}

namespace yaml {

#define BCase(X) IO.bitSetCase(Value, #X, COFF::X);

template <>
struct ScalarBitSetTraits<COFF::SectionCharacteristics> {
  static void bitset(IO &IO, COFF::SectionCharacteristics &Value) {
    BCase(IMAGE_SCN_TYPE_NO_PAD);
    BCase(IMAGE_SCN_CNT_CODE);
    BCase(IMAGE_SCN_CNT_INITIALIZED_DATA);
    BCase(IMAGE_SCN_CNT_UNINITIALIZED_DATA);
    BCase(IMAGE_SCN_LNK_OTHER);
    BCase(IMAGE_SCN_LNK_INFO);
    BCase(IMAGE_SCN_LNK_REMOVE);
    BCase(IMAGE_SCN_LNK_COMDAT);
    BCase(IMAGE_SCN_GPREL);
    BCase(IMAGE_SCN_MEM_PURGEABLE);
    BCase(IMAGE_SCN_MEM_16BIT);
    BCase(IMAGE_SCN_MEM_LOCKED);
    BCase(IMAGE_SCN_MEM_PRELOAD);
    BCase(IMAGE_SCN_ALIGN_1BYTES);
    BCase(IMAGE_SCN_ALIGN_2BYTES);
    BCase(IMAGE_SCN_ALIGN_4BYTES);
    BCase(IMAGE_SCN_ALIGN_8BYTES);
    BCase(IMAGE_SCN_ALIGN_16BYTES);
    BCase(IMAGE_SCN_ALIGN_32BYTES);
    BCase(IMAGE_SCN_ALIGN_64BYTES);
    BCase(IMAGE_SCN_ALIGN_128BYTES);
    BCase(IMAGE_SCN_ALIGN_256BYTES);
    BCase(IMAGE_SCN_ALIGN_512BYTES);
    BCase(IMAGE_SCN_ALIGN_1024BYTES);
    BCase(IMAGE_SCN_ALIGN_2048BYTES);
    BCase(IMAGE_SCN_ALIGN_4096BYTES);
    BCase(IMAGE_SCN_ALIGN_8192BYTES);
    BCase(IMAGE_SCN_LNK_NRELOC_OVFL);
    BCase(IMAGE_SCN_MEM_DISCARDABLE);
    BCase(IMAGE_SCN_MEM_NOT_CACHED);
    BCase(IMAGE_SCN_MEM_NOT_PAGED);
    BCase(IMAGE_SCN_MEM_SHARED);
    BCase(IMAGE_SCN_MEM_EXECUTE);
    BCase(IMAGE_SCN_MEM_READ);
    BCase(IMAGE_SCN_MEM_WRITE);
  }
};

template <>
struct ScalarBitSetTraits<COFF::Characteristics> {
  static void bitset(IO &IO, COFF::Characteristics &Value) {
    BCase(IMAGE_FILE_RELOCS_STRIPPED);
    BCase(IMAGE_FILE_EXECUTABLE_IMAGE);
    BCase(IMAGE_FILE_LINE_NUMS_STRIPPED);
    BCase(IMAGE_FILE_LOCAL_SYMS_STRIPPED);
    BCase(IMAGE_FILE_AGGRESSIVE_WS_TRIM);
    BCase(IMAGE_FILE_LARGE_ADDRESS_AWARE);
    BCase(IMAGE_FILE_BYTES_REVERSED_LO);
    BCase(IMAGE_FILE_32BIT_MACHINE);
    BCase(IMAGE_FILE_DEBUG_STRIPPED);
    BCase(IMAGE_FILE_REMOVABLE_RUN_FROM_SWAP);
    BCase(IMAGE_FILE_NET_RUN_FROM_SWAP);
    BCase(IMAGE_FILE_SYSTEM);
    BCase(IMAGE_FILE_DLL);
    BCase(IMAGE_FILE_UP_SYSTEM_ONLY);
    BCase(IMAGE_FILE_BYTES_REVERSED_HI);
  }
};
#undef BCase

#define ECase(X) IO.enumCase(Value, #X, COFF::X);

template <>
struct ScalarEnumerationTraits<COFF::SymbolComplexType> {
  static void enumeration(IO &IO, COFF::SymbolComplexType &Value) {
    ECase(IMAGE_SYM_DTYPE_NULL);
    ECase(IMAGE_SYM_DTYPE_POINTER);
    ECase(IMAGE_SYM_DTYPE_FUNCTION);
    ECase(IMAGE_SYM_DTYPE_ARRAY);
  }
};

template <>
struct ScalarEnumerationTraits<COFF::SymbolStorageClass> {
  static void enumeration(IO &IO, COFF::SymbolStorageClass &Value) {
    ECase(IMAGE_SYM_CLASS_END_OF_FUNCTION);
    ECase(IMAGE_SYM_CLASS_NULL);
    ECase(IMAGE_SYM_CLASS_AUTOMATIC);
    ECase(IMAGE_SYM_CLASS_EXTERNAL);
    ECase(IMAGE_SYM_CLASS_STATIC);
    ECase(IMAGE_SYM_CLASS_REGISTER);
    ECase(IMAGE_SYM_CLASS_EXTERNAL_DEF);
    ECase(IMAGE_SYM_CLASS_LABEL);
    ECase(IMAGE_SYM_CLASS_UNDEFINED_LABEL);
    ECase(IMAGE_SYM_CLASS_MEMBER_OF_STRUCT);
    ECase(IMAGE_SYM_CLASS_ARGUMENT);
    ECase(IMAGE_SYM_CLASS_STRUCT_TAG);
    ECase(IMAGE_SYM_CLASS_MEMBER_OF_UNION);
    ECase(IMAGE_SYM_CLASS_UNION_TAG);
    ECase(IMAGE_SYM_CLASS_TYPE_DEFINITION);
    ECase(IMAGE_SYM_CLASS_UNDEFINED_STATIC);
    ECase(IMAGE_SYM_CLASS_ENUM_TAG);
    ECase(IMAGE_SYM_CLASS_MEMBER_OF_ENUM);
    ECase(IMAGE_SYM_CLASS_REGISTER_PARAM);
    ECase(IMAGE_SYM_CLASS_BIT_FIELD);
    ECase(IMAGE_SYM_CLASS_BLOCK);
    ECase(IMAGE_SYM_CLASS_FUNCTION);
    ECase(IMAGE_SYM_CLASS_END_OF_STRUCT);
    ECase(IMAGE_SYM_CLASS_FILE);
    ECase(IMAGE_SYM_CLASS_SECTION);
    ECase(IMAGE_SYM_CLASS_WEAK_EXTERNAL);
    ECase(IMAGE_SYM_CLASS_CLR_TOKEN);
  }
};

template <>
struct ScalarEnumerationTraits<COFF::SymbolBaseType> {
  static void enumeration(IO &IO, COFF::SymbolBaseType &Value) {
    ECase(IMAGE_SYM_TYPE_NULL);
    ECase(IMAGE_SYM_TYPE_VOID);
    ECase(IMAGE_SYM_TYPE_CHAR);
    ECase(IMAGE_SYM_TYPE_SHORT);
    ECase(IMAGE_SYM_TYPE_INT);
    ECase(IMAGE_SYM_TYPE_LONG);
    ECase(IMAGE_SYM_TYPE_FLOAT);
    ECase(IMAGE_SYM_TYPE_DOUBLE);
    ECase(IMAGE_SYM_TYPE_STRUCT);
    ECase(IMAGE_SYM_TYPE_UNION);
    ECase(IMAGE_SYM_TYPE_ENUM);
    ECase(IMAGE_SYM_TYPE_MOE);
    ECase(IMAGE_SYM_TYPE_BYTE);
    ECase(IMAGE_SYM_TYPE_WORD);
    ECase(IMAGE_SYM_TYPE_UINT);
    ECase(IMAGE_SYM_TYPE_DWORD);
  }
};

template <>
struct ScalarEnumerationTraits<COFF::MachineTypes> {
  static void enumeration(IO &IO, COFF::MachineTypes &Value) {
    ECase(IMAGE_FILE_MACHINE_UNKNOWN);
    ECase(IMAGE_FILE_MACHINE_AM33);
    ECase(IMAGE_FILE_MACHINE_AMD64);
    ECase(IMAGE_FILE_MACHINE_ARM);
    ECase(IMAGE_FILE_MACHINE_ARMV7);
    ECase(IMAGE_FILE_MACHINE_EBC);
    ECase(IMAGE_FILE_MACHINE_I386);
    ECase(IMAGE_FILE_MACHINE_IA64);
    ECase(IMAGE_FILE_MACHINE_M32R);
    ECase(IMAGE_FILE_MACHINE_MIPS16);
    ECase(IMAGE_FILE_MACHINE_MIPSFPU);
    ECase(IMAGE_FILE_MACHINE_MIPSFPU16);
    ECase(IMAGE_FILE_MACHINE_POWERPC);
    ECase(IMAGE_FILE_MACHINE_POWERPCFP);
    ECase(IMAGE_FILE_MACHINE_R4000);
    ECase(IMAGE_FILE_MACHINE_SH3);
    ECase(IMAGE_FILE_MACHINE_SH3DSP);
    ECase(IMAGE_FILE_MACHINE_SH4);
    ECase(IMAGE_FILE_MACHINE_SH5);
    ECase(IMAGE_FILE_MACHINE_THUMB);
    ECase(IMAGE_FILE_MACHINE_WCEMIPSV2);
  }
};

template <>
struct ScalarEnumerationTraits<COFF::RelocationTypeX86> {
  static void enumeration(IO &IO, COFF::RelocationTypeX86 &Value) {
    ECase(IMAGE_REL_I386_ABSOLUTE);
    ECase(IMAGE_REL_I386_DIR16);
    ECase(IMAGE_REL_I386_REL16);
    ECase(IMAGE_REL_I386_DIR32);
    ECase(IMAGE_REL_I386_DIR32NB);
    ECase(IMAGE_REL_I386_SEG12);
    ECase(IMAGE_REL_I386_SECTION);
    ECase(IMAGE_REL_I386_SECREL);
    ECase(IMAGE_REL_I386_TOKEN);
    ECase(IMAGE_REL_I386_SECREL7);
    ECase(IMAGE_REL_I386_REL32);
    ECase(IMAGE_REL_AMD64_ABSOLUTE);
    ECase(IMAGE_REL_AMD64_ADDR64);
    ECase(IMAGE_REL_AMD64_ADDR32);
    ECase(IMAGE_REL_AMD64_ADDR32NB);
    ECase(IMAGE_REL_AMD64_REL32);
    ECase(IMAGE_REL_AMD64_REL32_1);
    ECase(IMAGE_REL_AMD64_REL32_2);
    ECase(IMAGE_REL_AMD64_REL32_3);
    ECase(IMAGE_REL_AMD64_REL32_4);
    ECase(IMAGE_REL_AMD64_REL32_5);
    ECase(IMAGE_REL_AMD64_SECTION);
    ECase(IMAGE_REL_AMD64_SECREL);
    ECase(IMAGE_REL_AMD64_SECREL7);
    ECase(IMAGE_REL_AMD64_TOKEN);
    ECase(IMAGE_REL_AMD64_SREL32);
    ECase(IMAGE_REL_AMD64_PAIR);
    ECase(IMAGE_REL_AMD64_SSPAN32);
  }
};

#undef ECase

template <>
struct MappingTraits<COFFYAML::Symbol> {
  struct NStorageClass {
    NStorageClass(IO&) : StorageClass(COFF::SymbolStorageClass(0)) {
    }
    NStorageClass(IO&, uint8_t S) : StorageClass(COFF::SymbolStorageClass(S)) {
    }
    uint8_t denormalize(IO &) {
      return StorageClass;
    }

    COFF::SymbolStorageClass StorageClass;
  };

  static void mapping(IO &IO, COFFYAML::Symbol &S) {
    MappingNormalization<NStorageClass, uint8_t> NS(IO, S.Header.StorageClass);

    IO.mapRequired("SimpleType", S.SimpleType);
    IO.mapOptional("NumberOfAuxSymbols", S.Header.NumberOfAuxSymbols);
    IO.mapRequired("Name", S.Name);
    IO.mapRequired("StorageClass", NS->StorageClass);
    IO.mapOptional("AuxiliaryData", S.AuxiliaryData);
    IO.mapRequired("ComplexType", S.ComplexType);
    IO.mapRequired("Value", S.Header.Value);
    IO.mapRequired("SectionNumber", S.Header.SectionNumber);
  }
};

template <>
struct MappingTraits<COFF::header> {
  struct NMachine {
    NMachine(IO&) : Machine(COFF::MachineTypes(0)) {
    }
    NMachine(IO&, uint16_t M) : Machine(COFF::MachineTypes(M)) {
    }
    uint16_t denormalize(IO &) {
      return Machine;
    }
    COFF::MachineTypes Machine;
  };

  struct NCharacteristics {
    NCharacteristics(IO&) : Characteristics(COFF::Characteristics(0)) {
    }
    NCharacteristics(IO&, uint16_t C) :
      Characteristics(COFF::Characteristics(C)) {
    }
    uint16_t denormalize(IO &) {
      return Characteristics;
    }

    COFF::Characteristics Characteristics;
  };

  static void mapping(IO &IO, COFF::header &H) {
    MappingNormalization<NMachine, uint16_t> NM(IO, H.Machine);
    MappingNormalization<NCharacteristics, uint16_t> NC(IO, H.Characteristics);

    IO.mapRequired("Machine", NM->Machine);
    IO.mapOptional("Characteristics", NC->Characteristics);
  }
};

template <>
struct MappingTraits<COFF::relocation> {
  struct NType {
    NType(IO &) : Type(COFF::RelocationTypeX86(0)) {
    }
    NType(IO &, uint16_t T) : Type(COFF::RelocationTypeX86(T)) {
    }
    uint16_t denormalize(IO &) {
      return Type;
    }
    COFF::RelocationTypeX86 Type;
  };

  static void mapping(IO &IO, COFF::relocation &Rel) {
    MappingNormalization<NType, uint16_t> NT(IO, Rel.Type);

    IO.mapRequired("Type", NT->Type);
    IO.mapRequired("VirtualAddress", Rel.VirtualAddress);
    IO.mapRequired("SymbolTableIndex", Rel.SymbolTableIndex);
  }
};

template <>
struct MappingTraits<COFFYAML::Section> {
  struct NCharacteristics {
    NCharacteristics(IO &) : Characteristics(COFF::SectionCharacteristics(0)) {
    }
    NCharacteristics(IO &, uint32_t C) :
      Characteristics(COFF::SectionCharacteristics(C)) {
    }
    uint32_t denormalize(IO &) {
      return Characteristics;
    }
    COFF::SectionCharacteristics Characteristics;
  };

  static void mapping(IO &IO, COFFYAML::Section &Sec) {
    MappingNormalization<NCharacteristics, uint32_t> NC(IO,
                                                    Sec.Header.Characteristics);
    IO.mapOptional("Relocations", Sec.Relocations);
    IO.mapRequired("SectionData", Sec.SectionData);
    IO.mapRequired("Characteristics", NC->Characteristics);
    IO.mapRequired("Name", Sec.Name);
  }
};

template <>
struct MappingTraits<COFFYAML::Object> {
  static void mapping(IO &IO, COFFYAML::Object &Obj) {
    IO.mapRequired("sections", Obj.Sections);
    IO.mapRequired("header", Obj.Header);
    IO.mapRequired("symbols", Obj.Symbols);
  }
};
} // end namespace yaml
} // end namespace llvm

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  OwningPtr<MemoryBuffer> Buf;
  if (MemoryBuffer::getFileOrSTDIN(Input, Buf))
    return 1;

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
  if (!writeCOFF(CP, outs())) {
    errs() << "yaml2obj: Failed to write COFF file!\n";
    return 1;
  }
}
