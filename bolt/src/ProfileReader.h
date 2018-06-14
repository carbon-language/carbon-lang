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
private:

  /// Adjustments for basic samples profiles (without LBR).
  bool NormalizeByInsnCount{false};
  bool NormalizeByCalls{false};

  /// Binary profile in YAML format.
  yaml::bolt::BinaryProfile YamlBP;

  /// Map a function ID from a YAML profile to a BinaryFunction object.
  std::vector<BinaryFunction *> YamlProfileToFunction;

  /// To keep track of functions that have a matched profile before the profile
  /// is attributed.
  std::unordered_set<const BinaryFunction *> ProfiledFunctions;

  /// Populate \p Function profile with the one supplied in YAML format.
  bool parseFunctionProfile(BinaryFunction &Function,
                            const yaml::bolt::BinaryFunctionProfile &YamlBF);

  /// For LTO symbol resolution.
  /// Map a common LTO prefix to a list of YAML profiles matching the prefix.
  StringMap<std::vector<yaml::bolt::BinaryFunctionProfile *>> LTOCommonNameMap;

  /// Map a common LTO prefix to a set of binary functions.
  StringMap<std::unordered_set<const BinaryFunction *>>
                                                      LTOCommonNameFunctionMap;

  /// Strict matching of a name in a profile to its contents.
  StringMap<yaml::bolt::BinaryFunctionProfile *> ProfileNameToProfile;

  /// Initialize maps for profile matching.
  void buildNameMaps(std::map<uint64_t, BinaryFunction> &Functions);

  /// Update matched YAML -> BinaryFunction pair.
  void matchProfileToFunction(yaml::bolt::BinaryFunctionProfile &YamlBF,
                              BinaryFunction &BF) {
    if (YamlBF.Id >= YamlProfileToFunction.size())
      YamlProfileToFunction.resize(YamlBF.Id + 1);
    YamlProfileToFunction[YamlBF.Id] = &BF;
    YamlBF.Used = true;

    assert(!ProfiledFunctions.count(&BF) &&
           "function already has an assigned profile");
    ProfiledFunctions.emplace(&BF);
  }

  /// Check if the profile uses an event with a given \p Name.
  bool usesEvent(StringRef Name) const;

public:
  /// Read profile from a file and associate with a set of functions.
  std::error_code readProfile(const std::string &FileName,
                              std::map<uint64_t, BinaryFunction> &Functions);

};

}
}

#endif
