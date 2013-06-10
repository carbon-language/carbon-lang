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

// For now, hardcode 64 bits everywhere that 32 or 64 would be needed
// since 64-bit can hold 32-bit values too.
struct Header {
  ELF_ELFCLASS Class;
  ELF_ELFDATA Data;
  ELF_ET Type;
  ELF_EM Machine;
  llvm::yaml::Hex64 Entry;
};
struct Object {
  Header Header;
};

} // end namespace ELFYAML
} // end namespace llvm

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
struct MappingTraits<ELFYAML::Header> {
  static void mapping(IO &IO, ELFYAML::Header &Header);
};

template <>
struct MappingTraits<ELFYAML::Object> {
  static void mapping(IO &IO, ELFYAML::Object &Object);
};

} // end namespace yaml
} // end namespace llvm

#endif
