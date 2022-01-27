//===-- RISCVISAInfo.h - RISCV ISA Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RISCVISAINFO_H
#define LLVM_SUPPORT_RISCVISAINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <string>
#include <vector>

namespace llvm {
struct RISCVExtensionInfo {
  std::string ExtName;
  unsigned MajorVersion;
  unsigned MinorVersion;
};

class RISCVISAInfo {
public:
  RISCVISAInfo(const RISCVISAInfo &) = delete;
  RISCVISAInfo &operator=(const RISCVISAInfo &) = delete;

  static bool compareExtension(const std::string &LHS, const std::string &RHS);

  /// Helper class for OrderedExtensionMap.
  struct ExtensionComparator {
    bool operator()(const std::string &LHS, const std::string &RHS) const {
      return compareExtension(LHS, RHS);
    }
  };

  /// OrderedExtensionMap is std::map, it's specialized to keep entries
  /// in canonical order of extension.
  typedef std::map<std::string, RISCVExtensionInfo, ExtensionComparator>
      OrderedExtensionMap;

  /// Parse RISCV ISA info from arch string.
  static llvm::Expected<std::unique_ptr<RISCVISAInfo>>
  parseArchString(StringRef Arch, bool EnableExperimentalExtension,
                  bool ExperimentalExtensionVersionCheck = true);

  /// Parse RISCV ISA info from feature vector.
  static llvm::Expected<std::unique_ptr<RISCVISAInfo>>
  parseFeatures(unsigned XLen, const std::vector<std::string> &Features);

  /// Convert RISCV ISA info to a feature vector.
  void toFeatures(std::vector<StringRef> &Features,
                  std::function<StringRef(const Twine &)> StrAlloc) const;

  const OrderedExtensionMap &getExtensions() const { return Exts; };

  unsigned getXLen() const { return XLen; };
  unsigned getFLen() const { return FLen; };
  unsigned getMinVLen() const { return MinVLen; }
  unsigned getMaxELen() const { return MaxELen; }
  unsigned getMaxELenFp() const { return MaxELenFp; }

  bool hasExtension(StringRef Ext) const;
  std::string toString() const;
  std::vector<std::string> toFeatureVector() const;

  static bool isSupportedExtensionFeature(StringRef Ext);
  static bool isSupportedExtension(StringRef Ext);
  static bool isSupportedExtension(StringRef Ext, unsigned MajorVersion,
                                   unsigned MinorVersion);

private:
  RISCVISAInfo(unsigned XLen)
      : XLen(XLen), FLen(0), MinVLen(0), MaxELen(0), MaxELenFp(0) {}

  unsigned XLen;
  unsigned FLen;
  unsigned MinVLen;
  unsigned MaxELen, MaxELenFp;

  OrderedExtensionMap Exts;

  void addExtension(StringRef ExtName, unsigned MajorVersion,
                    unsigned MinorVersion);

  Error checkDependency();

  void updateImplication();
  void updateFLen();
  void updateMinVLen();
  void updateMaxELen();
};

} // namespace llvm

#endif
