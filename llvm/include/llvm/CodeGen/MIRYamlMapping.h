//===- MIRYAMLMapping.h - Describes the mapping between MIR and YAML ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The MIR serialization library is currently a work in progress. It can't
// serialize machine functions at this time.
//
// This file implements the mapping between various MIR data structures and
// their corresponding YAML representation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MIRYAMLMAPPING_H
#define LLVM_LIB_CODEGEN_MIRYAMLMAPPING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace yaml {

struct MachineFunction {
  StringRef Name;
};

template <> struct MappingTraits<MachineFunction> {
  static void mapping(IO &YamlIO, MachineFunction &MF) {
    YamlIO.mapRequired("name", MF.Name);
  }
};

} // end namespace yaml
} // end namespace llvm

#endif
