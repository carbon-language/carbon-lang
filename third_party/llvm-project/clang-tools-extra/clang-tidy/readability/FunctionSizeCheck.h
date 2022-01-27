//===--- FunctionSizeCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace readability {

/// Checks for large functions based on various metrics.
///
/// These options are supported:
///
///   * `LineThreshold` - flag functions exceeding this number of lines. The
///     default is `-1` (ignore the number of lines).
///   * `StatementThreshold` - flag functions exceeding this number of
///     statements. This may differ significantly from the number of lines for
///     macro-heavy code. The default is `800`.
///   * `BranchThreshold` - flag functions exceeding this number of control
///     statements. The default is `-1` (ignore the number of branches).
///   * `ParameterThreshold` - flag functions having a high number of
///     parameters. The default is `-1` (ignore the number of parameters).
///   * `NestingThreshold` - flag compound statements which create next nesting
///     level after `NestingThreshold`. This may differ significantly from the
///     expected value for macro-heavy code. The default is `-1` (ignore the
///     nesting level).
///   * `VariableThreshold` - flag functions having a high number of variable
///     declarations. The default is `-1` (ignore the number of variables).
class FunctionSizeCheck : public ClangTidyCheck {
public:
  FunctionSizeCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const unsigned LineThreshold;
  const unsigned StatementThreshold;
  const unsigned BranchThreshold;
  const unsigned ParameterThreshold;
  const unsigned NestingThreshold;
  const unsigned VariableThreshold;
};

} // namespace readability
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H
