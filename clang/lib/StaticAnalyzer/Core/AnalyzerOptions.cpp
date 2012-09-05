//===-- AnalyzerOptions.cpp - Analysis Engine Options -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains special accessors for analyzer configuration options
// with string representations.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;

bool
AnalyzerOptions::mayInlineCXXMemberFunction(CXXInlineableMemberKind K) const {
  if (IPAMode < Inlining)
    return false;

  if (!CXXMemberInliningMode) {
    static const char *ModeKey = "c++-inlining";
    std::string ModeStr = Config.lookup(ModeKey);

    CXXInlineableMemberKind &MutableMode =
      const_cast<CXXInlineableMemberKind &>(CXXMemberInliningMode);

    MutableMode = llvm::StringSwitch<CXXInlineableMemberKind>(ModeStr)
      .Case("", CIMK_MemberFunctions)
      .Case("constructors", CIMK_Constructors)
      .Case("destructors", CIMK_Destructors)
      .Case("none", CIMK_None)
      .Case("methods", CIMK_MemberFunctions)
      .Default(CXXInlineableMemberKind());

    if (!MutableMode) {
      // FIXME: We should emit a warning here about an unknown inlining kind,
      // but the AnalyzerOptions doesn't have access to a diagnostic engine.
      MutableMode = CIMK_None;
    }
  }

  return CXXMemberInliningMode >= K;
}

bool AnalyzerOptions::includeTemporaryDtorsInCFG() const {
  return !Config.lookup("cfg-temporary-dtors").empty();
}
