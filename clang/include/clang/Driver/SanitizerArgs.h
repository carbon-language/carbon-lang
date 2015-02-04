//===--- SanitizerArgs.h - Arguments for sanitizer tools  -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_DRIVER_SANITIZERARGS_H
#define LLVM_CLANG_DRIVER_SANITIZERARGS_H

#include "clang/Basic/Sanitizers.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include <string>
#include <vector>

namespace clang {
namespace driver {

class Driver;
class ToolChain;

class SanitizerArgs {
  SanitizerSet Sanitizers;
  SanitizerSet RecoverableSanitizers;

  std::vector<std::string> BlacklistFiles;
  int SanitizeCoverage;
  int MsanTrackOrigins;
  int AsanFieldPadding;
  bool AsanZeroBaseShadow;
  bool UbsanTrapOnError;
  bool AsanSharedRuntime;
  bool LinkCXXRuntimes;

 public:
  /// Parses the sanitizer arguments from an argument list.
  SanitizerArgs(const ToolChain &TC, const llvm::opt::ArgList &Args);

  bool needsAsanRt() const { return Sanitizers.has(SanitizerKind::Address); }
  bool needsSharedAsanRt() const { return AsanSharedRuntime; }
  bool needsTsanRt() const { return Sanitizers.has(SanitizerKind::Thread); }
  bool needsMsanRt() const { return Sanitizers.has(SanitizerKind::Memory); }
  bool needsLsanRt() const {
    return Sanitizers.has(SanitizerKind::Leak) &&
           !Sanitizers.has(SanitizerKind::Address);
  }
  bool needsUbsanRt() const;
  bool needsDfsanRt() const { return Sanitizers.has(SanitizerKind::DataFlow); }

  bool sanitizesVptr() const { return Sanitizers.has(SanitizerKind::Vptr); }
  bool requiresPIE() const;
  bool needsUnwindTables() const;
  bool linkCXXRuntimes() const { return LinkCXXRuntimes; }
  void addArgs(const llvm::opt::ArgList &Args,
               llvm::opt::ArgStringList &CmdArgs) const;

 private:
  void clear();
  bool getDefaultBlacklist(const Driver &D, std::string &BLPath);
};

}  // namespace driver
}  // namespace clang

#endif
