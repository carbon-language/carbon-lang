//===- COFFYAML.h - COFF YAMLIO implementation ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares classes for handling the YAML representation of COFF.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_COFFYAML_H
#define LLVM_OBJECT_COFFYAML_H


#include "llvm/Support/COFF.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {

namespace COFF {
inline Characteristics operator|(Characteristics a, Characteristics b) {
  uint32_t Ret = static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
  return static_cast<Characteristics>(Ret);
}

inline SectionCharacteristics operator|(SectionCharacteristics a,
                                        SectionCharacteristics b) {
  uint32_t Ret = static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
  return static_cast<SectionCharacteristics>(Ret);
}
}

// The structure of the yaml files is not an exact 1:1 match to COFF. In order
// to use yaml::IO, we use these structures which are closer to the source.
namespace COFFYAML {
  /// In an object file this is just a binary blob. In an yaml file it is an hex
  /// string. Using this avoid having to allocate temporary strings.
  /// FIXME: not COFF specific.
  class BinaryRef {
    ArrayRef<uint8_t> BinaryData;
    StringRef HexData;
    bool isBinary;
  public:
    BinaryRef(ArrayRef<uint8_t> BinaryData)
        : BinaryData(BinaryData), isBinary(true) {}
    BinaryRef(StringRef HexData) : HexData(HexData), isBinary(false) {}
    BinaryRef() : isBinary(false) {}
    StringRef getHex() const {
      assert(!isBinary);
      return HexData;
    }
    ArrayRef<uint8_t> getBinary() const {
      assert(isBinary);
      return BinaryData;
    }
  };

  struct Section {
    COFF::section Header;
    unsigned Alignment;
    BinaryRef SectionData;
    std::vector<COFF::relocation> Relocations;
    StringRef Name;
    Section();
  };

  struct Symbol {
    COFF::symbol Header;
    COFF::SymbolBaseType SimpleType;
    COFF::SymbolComplexType ComplexType;
    BinaryRef AuxiliaryData;
    StringRef Name;
    Symbol();
  };

  struct Object {
    COFF::header Header;
    std::vector<Section> Sections;
    std::vector<Symbol> Symbols;
    Object();
  };
}
}

LLVM_YAML_IS_SEQUENCE_VECTOR(COFFYAML::Section)
LLVM_YAML_IS_SEQUENCE_VECTOR(COFFYAML::Symbol)
LLVM_YAML_IS_SEQUENCE_VECTOR(COFF::relocation)

namespace llvm {
namespace yaml {

template<>
struct ScalarTraits<COFFYAML::BinaryRef> {
  static void output(const COFFYAML::BinaryRef &, void*, llvm::raw_ostream &);
  static StringRef input(StringRef, void*, COFFYAML::BinaryRef &);
};

template <>
struct ScalarEnumerationTraits<COFF::MachineTypes> {
  static void enumeration(IO &IO, COFF::MachineTypes &Value);
};

template <>
struct ScalarEnumerationTraits<COFF::SymbolBaseType> {
  static void enumeration(IO &IO, COFF::SymbolBaseType &Value);
};

template <>
struct ScalarEnumerationTraits<COFF::SymbolStorageClass> {
  static void enumeration(IO &IO, COFF::SymbolStorageClass &Value);
};

template <>
struct ScalarEnumerationTraits<COFF::SymbolComplexType> {
  static void enumeration(IO &IO, COFF::SymbolComplexType &Value);
};

template <>
struct ScalarEnumerationTraits<COFF::RelocationTypeX86> {
  static void enumeration(IO &IO, COFF::RelocationTypeX86 &Value);
};

template <>
struct ScalarBitSetTraits<COFF::Characteristics> {
  static void bitset(IO &IO, COFF::Characteristics &Value);
};

template <>
struct ScalarBitSetTraits<COFF::SectionCharacteristics> {
  static void bitset(IO &IO, COFF::SectionCharacteristics &Value);
};

template <>
struct MappingTraits<COFF::relocation> {
  static void mapping(IO &IO, COFF::relocation &Rel);
};

template <>
struct MappingTraits<COFF::header> {
  static void mapping(IO &IO, COFF::header &H);
};

template <>
struct MappingTraits<COFFYAML::Symbol> {
  static void mapping(IO &IO, COFFYAML::Symbol &S);
};

template <>
struct MappingTraits<COFFYAML::Section> {
  static void mapping(IO &IO, COFFYAML::Section &Sec);
};

template <>
struct MappingTraits<COFFYAML::Object> {
  static void mapping(IO &IO, COFFYAML::Object &Obj);
};

} // end namespace yaml
} // end namespace llvm

#endif
