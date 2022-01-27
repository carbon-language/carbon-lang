//===--- AvoidUnderscoreInGoogletestNameCheck.cpp - clang-tidy --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "AvoidUnderscoreInGoogletestNameCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {
namespace tidy {
namespace google {
namespace readability {

constexpr llvm::StringLiteral KDisabledTestPrefix = "DISABLED_";

// Determines whether the macro is a Googletest test macro.
static bool isGoogletestTestMacro(StringRef MacroName) {
  static const llvm::StringSet<> MacroNames = {"TEST", "TEST_F", "TEST_P",
                                               "TYPED_TEST", "TYPED_TEST_P"};
  return MacroNames.find(MacroName) != MacroNames.end();
}

namespace {

class AvoidUnderscoreInGoogletestNameCallback : public PPCallbacks {
public:
  AvoidUnderscoreInGoogletestNameCallback(
      Preprocessor *PP, AvoidUnderscoreInGoogletestNameCheck *Check)
      : PP(PP), Check(Check) {}

  // Detects expansions of the TEST, TEST_F, TEST_P, TYPED_TEST, TYPED_TEST_P
  // macros and checks that their arguments do not have any underscores.
  void MacroExpands(const Token &MacroNameToken,
                    const MacroDefinition &MacroDefinition, SourceRange Range,
                    const MacroArgs *Args) override {
    IdentifierInfo *NameIdentifierInfo = MacroNameToken.getIdentifierInfo();
    if (!NameIdentifierInfo)
      return;
    StringRef MacroName = NameIdentifierInfo->getName();
    if (!isGoogletestTestMacro(MacroName) || !Args ||
        Args->getNumMacroArguments() < 2)
      return;
    const Token *TestCaseNameToken = Args->getUnexpArgument(0);
    const Token *TestNameToken = Args->getUnexpArgument(1);
    if (!TestCaseNameToken || !TestNameToken)
      return;
    std::string TestCaseName = PP->getSpelling(*TestCaseNameToken);
    if (TestCaseName.find('_') != std::string::npos)
      Check->diag(TestCaseNameToken->getLocation(),
                  "avoid using \"_\" in test case name \"%0\" according to "
                  "Googletest FAQ")
          << TestCaseName;

    std::string TestNameMaybeDisabled = PP->getSpelling(*TestNameToken);
    StringRef TestName = TestNameMaybeDisabled;
    TestName.consume_front(KDisabledTestPrefix);
    if (TestName.contains('_'))
      Check->diag(TestNameToken->getLocation(),
                  "avoid using \"_\" in test name \"%0\" according to "
                  "Googletest FAQ")
          << TestName;
  }

private:
  Preprocessor *PP;
  AvoidUnderscoreInGoogletestNameCheck *Check;
};

} // namespace

void AvoidUnderscoreInGoogletestNameCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(
      std::make_unique<AvoidUnderscoreInGoogletestNameCallback>(PP, this));
}

} // namespace readability
} // namespace google
} // namespace tidy
} // namespace clang
