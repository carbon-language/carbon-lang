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

namespace {

template<class T>
typename llvm::enable_if_c<std::numeric_limits<T>::is_integer, bool>::type
getAs(const llvm::yaml::ScalarNode *SN, T &Result) {
  SmallString<4> Storage;
  StringRef Value = SN->getValue(Storage);
  if (Value.getAsInteger(0, Result))
    return false;
  return true;
}

// Given a container with begin and end with ::value_type of a character type.
// Iterate through pairs of characters in the the set of [a-fA-F0-9] ignoring
// all other characters.
struct hex_pair_iterator {
  StringRef::const_iterator Current, End;
  typedef SmallVector<char, 2> value_type;
  value_type Pair;
  bool IsDone;

  hex_pair_iterator(StringRef C)
    : Current(C.begin()), End(C.end()), IsDone(false) {
    // Initalize Pair.
    ++*this;
  }

  // End iterator.
  hex_pair_iterator() : Current(), End(), IsDone(true) {}

  value_type operator *() const {
    return Pair;
  }

  hex_pair_iterator operator ++() {
    // We're at the end of the input.
    if (Current == End) {
      IsDone = true;
      return *this;
    }
    Pair = value_type();
    for (; Current != End && Pair.size() != 2; ++Current) {
      // Is a valid hex digit.
      if ((*Current >= '0' && *Current <= '9') ||
          (*Current >= 'a' && *Current <= 'f') ||
          (*Current >= 'A' && *Current <= 'F'))
        Pair.push_back(*Current);
    }
    // Hit the end without getting 2 hex digits. Pair is invalid.
    if (Pair.size() != 2)
      IsDone = true;
    return *this;
  }

  bool operator ==(const hex_pair_iterator Other) {
    return (IsDone == Other.IsDone) ||
           (Current == Other.Current && End == Other.End);
  }

  bool operator !=(const hex_pair_iterator Other) {
    return !(*this == Other);
  }
};

template <class ContainerOut>
static bool hexStringToByteArray(StringRef Str, ContainerOut &Out) {
  for (hex_pair_iterator I(Str), E; I != E; ++I) {
    typename hex_pair_iterator::value_type Pair = *I;
    typename ContainerOut::value_type Byte;
    if (StringRef(Pair.data(), 2).getAsInteger(16, Byte))
      return false;
    Out.push_back(Byte);
  }
  return true;
}

// The structure of the yaml files is not an exact 1:1 match to COFF. In order
// to use yaml::IO, we use these structures which are closer to the source.
namespace COFFYAML {
  struct Relocation {
    uint32_t VirtualAddress;
    uint32_t SymbolTableIndex;
    COFF::RelocationTypeX86 Type;
  };

  struct Section {
    std::vector<COFF::SectionCharacteristics> Characteristics;
    StringRef SectionData;
    std::vector<Relocation> Relocations;
    StringRef Name;
  };

  struct Header {
    COFF::MachineTypes Machine;
  };

  struct Symbol {
    COFF::SymbolBaseType SimpleType;
    uint8_t NumberOfAuxSymbols;
    StringRef Name;
    COFF::SymbolStorageClass StorageClass;
    StringRef AuxillaryData;
    COFF::SymbolComplexType ComplexType;
    uint32_t Value;
    uint16_t SectionNumber;
  };

  struct Object {
    Header HeaderData;
    std::vector<Section> Sections;
    std::vector<Symbol> Symbols;
  };
}

/// This parses a yaml stream that represents a COFF object file.
/// See docs/yaml2obj for the yaml scheema.
struct COFFParser {
  COFFParser(COFFYAML::Object &Obj) : Obj(Obj) {
    std::memset(&Header, 0, sizeof(Header));
    // A COFF string table always starts with a 4 byte size field. Offsets into
    // it include this size, so allocate it now.
    StringTable.append(4, 0);
  }

  bool parseSections() {
    for (std::vector<COFFYAML::Section>::iterator i = Obj.Sections.begin(),
           e = Obj.Sections.end(); i != e; ++i) {
      const COFFYAML::Section &YamlSection = *i;
      Section Sec;
      std::memset(&Sec.Header, 0, sizeof(Sec.Header));

      // If the name is less than 8 bytes, store it in place, otherwise
      // store it in the string table.
      StringRef Name = YamlSection.Name;
      std::fill_n(Sec.Header.Name, unsigned(COFF::NameSize), 0);
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

      for (std::vector<COFF::SectionCharacteristics>::const_iterator i =
             YamlSection.Characteristics.begin(),
             e = YamlSection.Characteristics.end();
           i != e; ++i) {
        uint32_t Characteristic = *i;
        Sec.Header.Characteristics |= Characteristic;
      }

      StringRef Data = YamlSection.SectionData;
      if (!hexStringToByteArray(Data, Sec.Data)) {
        errs() << "SectionData must be a collection of pairs of hex bytes";
        return false;
      }
      Sections.push_back(Sec);
    }
    return true;
  }

  bool parseSymbols() {
    for (std::vector<COFFYAML::Symbol>::iterator i = Obj.Symbols.begin(),
           e = Obj.Symbols.end(); i != e; ++i) {
      COFFYAML::Symbol YamlSymbol = *i;
      Symbol Sym;
      std::memset(&Sym.Header, 0, sizeof(Sym.Header));

      // If the name is less than 8 bytes, store it in place, otherwise
      // store it in the string table.
      StringRef Name = YamlSymbol.Name;
      std::fill_n(Sym.Header.Name, unsigned(COFF::NameSize), 0);
      if (Name.size() <= COFF::NameSize) {
        std::copy(Name.begin(), Name.end(), Sym.Header.Name);
      } else {
        // Add string to the string table and format the index for output.
        unsigned Index = getStringIndex(Name);
        *reinterpret_cast<support::aligned_ulittle32_t*>(
            Sym.Header.Name + 4) = Index;
      }

      Sym.Header.Value = YamlSymbol.Value;
      Sym.Header.Type |= YamlSymbol.SimpleType;
      Sym.Header.Type |= YamlSymbol.ComplexType << COFF::SCT_COMPLEX_TYPE_SHIFT;
      Sym.Header.StorageClass = YamlSymbol.StorageClass;
      Sym.Header.SectionNumber = YamlSymbol.SectionNumber;

      StringRef Data = YamlSymbol.AuxillaryData;
      if (!hexStringToByteArray(Data, Sym.AuxSymbols)) {
        errs() << "AuxillaryData must be a collection of pairs of hex bytes";
        return false;
      }
      Symbols.push_back(Sym);
    }
    return true;
  }

  bool parse() {
    Header.Machine = Obj.HeaderData.Machine;
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
  COFF::header Header;

  struct Section {
    COFF::section Header;
    std::vector<uint8_t> Data;
    std::vector<COFF::relocation> Relocations;
  };

  struct Symbol {
    COFF::symbol Header;
    std::vector<uint8_t> AuxSymbols;
  };

  std::vector<Section> Sections;
  std::vector<Symbol> Symbols;
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
  SectionTableStart = sizeof(COFF::header) + CP.Header.SizeOfOptionalHeader;
  SectionTableSize = sizeof(COFF::section) * CP.Sections.size();

  uint32_t CurrentSectionDataOffset = SectionTableStart + SectionTableSize;

  // Assign each section data address consecutively.
  for (std::vector<COFFParser::Section>::iterator i = CP.Sections.begin(),
                                                  e = CP.Sections.end();
                                                  i != e; ++i) {
    if (!i->Data.empty()) {
      i->Header.SizeOfRawData = i->Data.size();
      i->Header.PointerToRawData = CurrentSectionDataOffset;
      CurrentSectionDataOffset += i->Header.SizeOfRawData;
      // TODO: Handle alignment.
    } else {
      i->Header.SizeOfRawData = 0;
      i->Header.PointerToRawData = 0;
    }
  }

  uint32_t SymbolTableStart = CurrentSectionDataOffset;

  // Calculate number of symbols.
  uint32_t NumberOfSymbols = 0;
  for (std::vector<COFFParser::Symbol>::iterator i = CP.Symbols.begin(),
                                                 e = CP.Symbols.end();
                                                 i != e; ++i) {
    if (i->AuxSymbols.size() % COFF::SymbolSize != 0) {
      errs() << "AuxillaryData size not a multiple of symbol size!\n";
      return false;
    }
    i->Header.NumberOfAuxSymbols = i->AuxSymbols.size() / COFF::SymbolSize;
    NumberOfSymbols += 1 + i->Header.NumberOfAuxSymbols;
  }

  // Store all the allocated start addresses in the header.
  CP.Header.NumberOfSections = CP.Sections.size();
  CP.Header.NumberOfSymbols = NumberOfSymbols;
  CP.Header.PointerToSymbolTable = SymbolTableStart;

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

void writeCOFF(COFFParser &CP, raw_ostream &OS) {
  OS << binary_le(CP.Header.Machine)
     << binary_le(CP.Header.NumberOfSections)
     << binary_le(CP.Header.TimeDateStamp)
     << binary_le(CP.Header.PointerToSymbolTable)
     << binary_le(CP.Header.NumberOfSymbols)
     << binary_le(CP.Header.SizeOfOptionalHeader)
     << binary_le(CP.Header.Characteristics);

  // Output section table.
  for (std::vector<COFFParser::Section>::const_iterator i = CP.Sections.begin(),
                                                        e = CP.Sections.end();
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
  for (std::vector<COFFParser::Section>::const_iterator i = CP.Sections.begin(),
                                                        e = CP.Sections.end();
                                                        i != e; ++i) {
    if (!i->Data.empty())
      OS.write(reinterpret_cast<const char*>(&i->Data[0]), i->Data.size());
  }

  // Output symbol table.

  for (std::vector<COFFParser::Symbol>::const_iterator i = CP.Symbols.begin(),
                                                       e = CP.Symbols.end();
                                                       i != e; ++i) {
    OS.write(i->Header.Name, COFF::NameSize);
    OS << binary_le(i->Header.Value)
       << binary_le(i->Header.SectionNumber)
       << binary_le(i->Header.Type)
       << binary_le(i->Header.StorageClass)
       << binary_le(i->Header.NumberOfAuxSymbols);
    if (!i->AuxSymbols.empty())
      OS.write( reinterpret_cast<const char*>(&i->AuxSymbols[0])
              , i->AuxSymbols.size());
  }

  // Output string table.
  OS.write(&CP.StringTable[0], CP.StringTable.size());
}

}

LLVM_YAML_IS_SEQUENCE_VECTOR(COFFYAML::Relocation)
LLVM_YAML_IS_SEQUENCE_VECTOR(COFF::SectionCharacteristics)
LLVM_YAML_IS_SEQUENCE_VECTOR(COFFYAML::Section)
LLVM_YAML_IS_SEQUENCE_VECTOR(COFFYAML::Symbol)

namespace llvm {
namespace yaml {
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

// FIXME: We cannot use ScalarBitSetTraits because of
// IMAGE_SYM_CLASS_END_OF_FUNCTION which is -1.
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
struct ScalarEnumerationTraits<COFF::SectionCharacteristics> {
  static void enumeration(IO &IO, COFF::SectionCharacteristics &Value) {
    ECase(IMAGE_SCN_TYPE_NO_PAD);
    ECase(IMAGE_SCN_CNT_CODE);
    ECase(IMAGE_SCN_CNT_INITIALIZED_DATA);
    ECase(IMAGE_SCN_CNT_UNINITIALIZED_DATA);
    ECase(IMAGE_SCN_LNK_OTHER);
    ECase(IMAGE_SCN_LNK_INFO);
    ECase(IMAGE_SCN_LNK_REMOVE);
    ECase(IMAGE_SCN_LNK_COMDAT);
    ECase(IMAGE_SCN_GPREL);
    ECase(IMAGE_SCN_MEM_PURGEABLE);
    ECase(IMAGE_SCN_MEM_16BIT);
    ECase(IMAGE_SCN_MEM_LOCKED);
    ECase(IMAGE_SCN_MEM_PRELOAD);
    ECase(IMAGE_SCN_ALIGN_1BYTES);
    ECase(IMAGE_SCN_ALIGN_2BYTES);
    ECase(IMAGE_SCN_ALIGN_4BYTES);
    ECase(IMAGE_SCN_ALIGN_8BYTES);
    ECase(IMAGE_SCN_ALIGN_16BYTES);
    ECase(IMAGE_SCN_ALIGN_32BYTES);
    ECase(IMAGE_SCN_ALIGN_64BYTES);
    ECase(IMAGE_SCN_ALIGN_128BYTES);
    ECase(IMAGE_SCN_ALIGN_256BYTES);
    ECase(IMAGE_SCN_ALIGN_512BYTES);
    ECase(IMAGE_SCN_ALIGN_1024BYTES);
    ECase(IMAGE_SCN_ALIGN_2048BYTES);
    ECase(IMAGE_SCN_ALIGN_4096BYTES);
    ECase(IMAGE_SCN_ALIGN_8192BYTES);
    ECase(IMAGE_SCN_LNK_NRELOC_OVFL);
    ECase(IMAGE_SCN_MEM_DISCARDABLE);
    ECase(IMAGE_SCN_MEM_NOT_CACHED);
    ECase(IMAGE_SCN_MEM_NOT_PAGED);
    ECase(IMAGE_SCN_MEM_SHARED);
    ECase(IMAGE_SCN_MEM_EXECUTE);
    ECase(IMAGE_SCN_MEM_READ);
    ECase(IMAGE_SCN_MEM_WRITE);
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
  static void mapping(IO &IO, COFFYAML::Symbol &S) {
    IO.mapRequired("SimpleType", S.SimpleType);
    IO.mapOptional("NumberOfAuxSymbols", S.NumberOfAuxSymbols);
    IO.mapRequired("Name", S.Name);
    IO.mapRequired("StorageClass", S.StorageClass);
    IO.mapOptional("AuxillaryData", S.AuxillaryData); // FIXME: typo
    IO.mapRequired("ComplexType", S.ComplexType);
    IO.mapRequired("Value", S.Value);
    IO.mapRequired("SectionNumber", S.SectionNumber);
  }
};

template <>
struct MappingTraits<COFFYAML::Header> {
  static void mapping(IO &IO, COFFYAML::Header &H) {
    IO.mapRequired("Machine", H.Machine);
  }
};

template <>
struct MappingTraits<COFFYAML::Relocation> {
  static void mapping(IO &IO, COFFYAML::Relocation &Rel) {
    IO.mapRequired("Type", Rel.Type);
    IO.mapRequired("VirtualAddress", Rel.VirtualAddress);
    IO.mapRequired("SymbolTableIndex", Rel.SymbolTableIndex);
  }
};

template <>
struct MappingTraits<COFFYAML::Section> {
  static void mapping(IO &IO, COFFYAML::Section &Sec) {
    IO.mapOptional("Relocations", Sec.Relocations);
    IO.mapRequired("SectionData", Sec.SectionData);
    IO.mapRequired("Characteristics", Sec.Characteristics);
    IO.mapRequired("Name", Sec.Name);
  }
};

template <>
struct MappingTraits<COFFYAML::Object> {
  static void mapping(IO &IO, COFFYAML::Object &Obj) {
    IO.mapRequired("sections", Obj.Sections);
    IO.mapRequired("header", Obj.HeaderData);
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
  writeCOFF(CP, outs());
}
