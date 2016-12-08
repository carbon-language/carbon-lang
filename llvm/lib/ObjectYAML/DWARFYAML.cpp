//===- DWARFYAML.cpp - DWARF YAMLIO implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of DWARF Debug
// Info.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFYAML.h"

namespace llvm {

bool DWARFYAML::Data::isEmpty() const {
  return 0 == DebugStrings.size() + AbbrevDecls.size();
}

namespace yaml {

void MappingTraits<DWARFYAML::Data>::mapping(
    IO &IO, DWARFYAML::Data &DWARF) {
  IO.mapOptional("debug_str", DWARF.DebugStrings);
  IO.mapOptional("debug_abbrev", DWARF.AbbrevDecls);
}

void MappingTraits<DWARFYAML::Abbrev>::mapping(
    IO &IO, DWARFYAML::Abbrev &Abbrev) {
  IO.mapRequired("Code", Abbrev.Code);
  IO.mapRequired("Tag", Abbrev.Tag);
  IO.mapRequired("Children", Abbrev.Children);
  IO.mapRequired("Attributes", Abbrev.Attributes);
}

void MappingTraits<DWARFYAML::AttributeAbbrev>::mapping(
    IO &IO, DWARFYAML::AttributeAbbrev &AttAbbrev) {
  IO.mapRequired("Attribute", AttAbbrev.Attribute);
  IO.mapRequired("Form", AttAbbrev.Form);
}

} // namespace llvm::yaml

} // namespace llvm
