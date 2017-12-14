//===-- ProfileReader.h - BOLT profile deserializer -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_PROFILEREADER_H
#define LLVM_TOOLS_LLVM_BOLT_PROFILEREADER_H

#include "BinaryFunction.h"
#include "ProfileYAMLMapping.h"
#include <unordered_set>

namespace llvm {
namespace bolt {

class ProfileReader {
  /// Number of function profiles that were unused by the reader.
  uint64_t NumUnusedProfiles{0};

  /// Map a function ID from a profile to a BinaryFunction object.
  std::vector<BinaryFunction *> YamlProfileToFunction;

  void reportError(StringRef Message);

  bool parseFunctionProfile(BinaryFunction &Function,
                            const yaml::bolt::BinaryFunctionProfile &YamlBF);

  /// Profile for binary functions.
  std::vector<yaml::bolt::BinaryFunctionProfile> YamlBFs;

  /// For LTO symbol resolution.
  /// Map a common LTO prefix to a list of profiles matching the prefix.
  StringMap<std::vector<yaml::bolt::BinaryFunctionProfile *>> LTOCommonNameMap;

  /// Map a common LTO prefix to a set of binary functions.
  StringMap<std::unordered_set<const BinaryFunction *>>
                                                      LTOCommonNameFunctionMap;

  StringMap<yaml::bolt::BinaryFunctionProfile *> ProfileNameToProfile;

  void buildNameMaps(std::map<uint64_t, BinaryFunction> &Functions);

  /// Update matched YAML -> BinaryFunction pair.
  void matchProfileToFunction(yaml::bolt::BinaryFunctionProfile &YamlBF,
                              BinaryFunction &BF) {
    if (YamlBF.Id >= YamlProfileToFunction.size())
      YamlProfileToFunction.resize(YamlBF.Id + 1);
    YamlProfileToFunction[YamlBF.Id] = &BF;
    YamlBF.Used = true;
  }

public:
  /// Read profile from a file and associate with a set of functions.
  std::error_code readProfile(const std::string &FileName,
                              std::map<uint64_t, BinaryFunction> &Functions);

};

}
}

#endif
