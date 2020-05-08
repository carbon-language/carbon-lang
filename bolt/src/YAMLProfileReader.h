//===-- YAMLProfileReader.h - BOLT YAML profile deserializer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_YAML_PROFILE_READER_H
#define LLVM_TOOLS_LLVM_BOLT_YAML_PROFILE_READER_H

#include "BinaryFunction.h"
#include "ProfileReaderBase.h"
#include "ProfileYAMLMapping.h"
#include <unordered_set>

namespace llvm {
namespace bolt {

class YAMLProfileReader : public ProfileReaderBase {
public:
  explicit YAMLProfileReader(StringRef Filename)
    : ProfileReaderBase(Filename) {}

  StringRef getReaderName() const override {
    return "YAML profile reader";
  }

  bool isTrustedSource() const override {
    return false;
  }

  Error readProfilePreCFG(BinaryContext &BC) override {
    return Error::success();
  }

  Error readProfile(BinaryContext &BC) override;

  Error preprocessProfile(BinaryContext &BC) override;

  virtual bool hasLocalsWithFileName() const override;

  virtual bool mayHaveProfileData(const BinaryFunction &BF) override;

  /// Check if the file contains YAML.
  static bool isYAML(StringRef Filename);

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

  /// For LTO symbol resolution.
  /// Map a common LTO prefix to a list of YAML profiles matching the prefix.
  StringMap<std::vector<yaml::bolt::BinaryFunctionProfile *>> LTOCommonNameMap;

  /// Map a common LTO prefix to a set of binary functions.
  StringMap<std::unordered_set<const BinaryFunction *>>
                                                      LTOCommonNameFunctionMap;

  /// Strict matching of a name in a profile to its contents.
  StringMap<yaml::bolt::BinaryFunctionProfile *> ProfileNameToProfile;

  /// Populate \p Function profile with the one supplied in YAML format.
  bool parseFunctionProfile(BinaryFunction &Function,
                            const yaml::bolt::BinaryFunctionProfile &YamlBF);

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
};

}
}

#endif
