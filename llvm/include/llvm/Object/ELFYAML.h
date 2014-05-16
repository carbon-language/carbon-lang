//===- ELFYAML.h - ELF YAMLIO implementation --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares classes for handling the YAML representation
/// of ELF.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELFYAML_H
#define LLVM_OBJECT_ELFYAML_H

#include "llvm/Object/YAML.h"
#include "llvm/Support/ELF.h"

namespace llvm {
namespace ELFYAML {

// These types are invariant across 32/64-bit ELF, so for simplicity just
// directly give them their exact sizes. We don't need to worry about
// endianness because these are just the types in the YAMLIO structures,
// and are appropriately converted to the necessary endianness when
// reading/generating binary object files.
// The naming of these types is intended to be ELF_PREFIX, where PREFIX is
// the common prefix of the respective constants. E.g. ELF_EM corresponds
// to the `e_machine` constants, like `EM_X86_64`.
// In the future, these would probably be better suited by C++11 enum
// class's with appropriate fixed underlying type.
LLVM_YAML_STRONG_TYPEDEF(uint16_t, ELF_ET)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, ELF_EM)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_ELFCLASS)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_ELFDATA)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_ELFOSABI)
// Just use 64, since it can hold 32-bit values too.
LLVM_YAML_STRONG_TYPEDEF(uint64_t, ELF_EF)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, ELF_SHT)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_REL)
// Just use 64, since it can hold 32-bit values too.
LLVM_YAML_STRONG_TYPEDEF(uint64_t, ELF_SHF)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ELF_STT)

// For now, hardcode 64 bits everywhere that 32 or 64 would be needed
// since 64-bit can hold 32-bit values too.
struct FileHeader {
  ELF_ELFCLASS Class;
  ELF_ELFDATA Data;
  ELF_ELFOSABI OSABI;
  ELF_ET Type;
  ELF_EM Machine;
  ELF_EF Flags;
  llvm::yaml::Hex64 Entry;
};
struct Symbol {
  StringRef Name;
  ELF_STT Type;
  StringRef Section;
  llvm::yaml::Hex64 Value;
  llvm::yaml::Hex64 Size;
};
struct LocalGlobalWeakSymbols {
  std::vector<Symbol> Local;
  std::vector<Symbol> Global;
  std::vector<Symbol> Weak;
};
struct Section {
  enum class SectionKind { RawContent, Relocation };
  SectionKind Kind;
  StringRef Name;
  ELF_SHT Type;
  ELF_SHF Flags;
  llvm::yaml::Hex64 Address;
  StringRef Link;
  StringRef Info;
  llvm::yaml::Hex64 AddressAlign;
  Section(SectionKind Kind) : Kind(Kind) {}
  virtual ~Section();
};
struct RawContentSection : Section {
  object::yaml::BinaryRef Content;
  llvm::yaml::Hex64 Size;
  RawContentSection() : Section(SectionKind::RawContent) {}
  static bool classof(const Section *S) {
    return S->Kind == SectionKind::RawContent;
  }
};
struct Relocation {
  llvm::yaml::Hex64 Offset;
  int64_t Addend;
  ELF_REL Type;
  StringRef Symbol;
};
struct RelocationSection : Section {
  std::vector<Relocation> Relocations;
  RelocationSection() : Section(SectionKind::Relocation) {}
  static bool classof(const Section *S) {
    return S->Kind == SectionKind::Relocation;
  }
};
struct Object {
  FileHeader Header;
  std::vector<std::unique_ptr<Section>> Sections;
  // Although in reality the symbols reside in a section, it is a lot
  // cleaner and nicer if we read them from the YAML as a separate
  // top-level key, which automatically ensures that invariants like there
  // being a single SHT_SYMTAB section are upheld.
  LocalGlobalWeakSymbols Symbols;
};

} // end namespace ELFYAML
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(std::unique_ptr<llvm::ELFYAML::Section>)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::Symbol)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::ELFYAML::Relocation)

namespace llvm {
namespace yaml {

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_ET> {
  static void enumeration(IO &IO, ELFYAML::ELF_ET &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_EM> {
  static void enumeration(IO &IO, ELFYAML::ELF_EM &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_ELFCLASS> {
  static void enumeration(IO &IO, ELFYAML::ELF_ELFCLASS &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_ELFDATA> {
  static void enumeration(IO &IO, ELFYAML::ELF_ELFDATA &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_ELFOSABI> {
  static void enumeration(IO &IO, ELFYAML::ELF_ELFOSABI &Value);
};

template <>
struct ScalarBitSetTraits<ELFYAML::ELF_EF> {
  static void bitset(IO &IO, ELFYAML::ELF_EF &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_SHT> {
  static void enumeration(IO &IO, ELFYAML::ELF_SHT &Value);
};

template <>
struct ScalarBitSetTraits<ELFYAML::ELF_SHF> {
  static void bitset(IO &IO, ELFYAML::ELF_SHF &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_STT> {
  static void enumeration(IO &IO, ELFYAML::ELF_STT &Value);
};

template <>
struct ScalarEnumerationTraits<ELFYAML::ELF_REL> {
  static void enumeration(IO &IO, ELFYAML::ELF_REL &Value);
};

template <>
struct MappingTraits<ELFYAML::FileHeader> {
  static void mapping(IO &IO, ELFYAML::FileHeader &FileHdr);
};

template <>
struct MappingTraits<ELFYAML::Symbol> {
  static void mapping(IO &IO, ELFYAML::Symbol &Symbol);
};

template <>
struct MappingTraits<ELFYAML::LocalGlobalWeakSymbols> {
  static void mapping(IO &IO, ELFYAML::LocalGlobalWeakSymbols &Symbols);
};

template <> struct MappingTraits<ELFYAML::Relocation> {
  static void mapping(IO &IO, ELFYAML::Relocation &Rel);
};

template <>
struct MappingTraits<std::unique_ptr<ELFYAML::Section>> {
  static void mapping(IO &IO, std::unique_ptr<ELFYAML::Section> &Section);
  static StringRef validate(IO &io, std::unique_ptr<ELFYAML::Section> &Section);
};

template <>
struct MappingTraits<ELFYAML::Object> {
  static void mapping(IO &IO, ELFYAML::Object &Object);
};

} // end namespace yaml
} // end namespace llvm

#endif
