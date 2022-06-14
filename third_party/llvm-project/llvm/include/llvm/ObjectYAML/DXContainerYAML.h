//===- DXContainerYAML.h - DXContainer YAMLIO implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares classes for handling the YAML representation
/// of DXContainer.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_DXCONTAINERYAML_H
#define LLVM_OBJECTYAML_DXCONTAINERYAML_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/YAMLTraits.h"
#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
namespace DXContainerYAML {

struct VersionTuple {
  uint16_t Major;
  uint16_t Minor;
};

// The optional header fields are required in the binary and will be populated
// when reading from binary, but can be omitted in the YAML text because the
// emitter can calculate them.
struct FileHeader {
  std::vector<llvm::yaml::Hex8> Hash;
  VersionTuple Version;
  Optional<uint32_t> FileSize;
  uint32_t PartCount;
  Optional<std::vector<uint32_t>> PartOffsets;
};

struct DXILProgram {
  uint8_t MajorVersion;
  uint8_t MinorVersion;
  uint16_t ShaderKind;
  Optional<uint32_t> Size;
  uint16_t DXILMajorVersion;
  uint16_t DXILMinorVersion;
  Optional<uint32_t> DXILOffset;
  Optional<uint32_t> DXILSize;
  Optional<std::vector<llvm::yaml::Hex8>> DXIL;
};

struct Part {
  std::string Name;
  uint32_t Size;
  Optional<DXILProgram> Program;
};

struct Object {
  FileHeader Header;
  std::vector<Part> Parts;
};

} // namespace DXContainerYAML
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::DXContainerYAML::Part)
namespace llvm {

class raw_ostream;

namespace yaml {

template <> struct MappingTraits<DXContainerYAML::VersionTuple> {
  static void mapping(IO &IO, DXContainerYAML::VersionTuple &Version);
};

template <> struct MappingTraits<DXContainerYAML::FileHeader> {
  static void mapping(IO &IO, DXContainerYAML::FileHeader &Header);
};

template <> struct MappingTraits<DXContainerYAML::DXILProgram> {
  static void mapping(IO &IO, DXContainerYAML::DXILProgram &Program);
};

template <> struct MappingTraits<DXContainerYAML::Part> {
  static void mapping(IO &IO, DXContainerYAML::Part &Version);
};

template <> struct MappingTraits<DXContainerYAML::Object> {
  static void mapping(IO &IO, DXContainerYAML::Object &Obj);
};

} // namespace yaml

} // namespace llvm

#endif // LLVM_OBJECTYAML_DXCONTAINERYAML_H
