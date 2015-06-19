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
#include <vector>

namespace llvm {
namespace yaml {

struct MachineBasicBlock {
  std::string Name;
  unsigned Alignment = 0;
  bool IsLandingPad = false;
  bool AddressTaken = false;
  // TODO: Serialize the successors and liveins.
  // TODO: Serialize machine instructions.
};

template <> struct MappingTraits<MachineBasicBlock> {
  static void mapping(IO &YamlIO, MachineBasicBlock &MBB) {
    YamlIO.mapOptional("name", MBB.Name,
                       std::string()); // Don't print out an empty name.
    YamlIO.mapOptional("alignment", MBB.Alignment);
    YamlIO.mapOptional("isLandingPad", MBB.IsLandingPad);
    YamlIO.mapOptional("addressTaken", MBB.AddressTaken);
  }
};

} // end namespace yaml
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::yaml::MachineBasicBlock)

namespace llvm {
namespace yaml {

struct MachineFunction {
  StringRef Name;
  unsigned Alignment = 0;
  bool ExposesReturnsTwice = false;
  bool HasInlineAsm = false;

  std::vector<MachineBasicBlock> BasicBlocks;
};

template <> struct MappingTraits<MachineFunction> {
  static void mapping(IO &YamlIO, MachineFunction &MF) {
    YamlIO.mapRequired("name", MF.Name);
    YamlIO.mapOptional("alignment", MF.Alignment);
    YamlIO.mapOptional("exposesReturnsTwice", MF.ExposesReturnsTwice);
    YamlIO.mapOptional("hasInlineAsm", MF.HasInlineAsm);
    YamlIO.mapOptional("body", MF.BasicBlocks);
  }
};

} // end namespace yaml
} // end namespace llvm

#endif
