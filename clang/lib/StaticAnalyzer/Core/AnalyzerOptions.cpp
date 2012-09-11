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
using namespace llvm;

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

bool AnalyzerOptions::getBooleanOption(StringRef Name, bool DefaultVal) const {
  // FIXME: We should emit a warning here if the value is something other than
  // "true", "false", or the empty string (meaning the default value),
  // but the AnalyzerOptions doesn't have access to a diagnostic engine.
  return llvm::StringSwitch<bool>(Config.lookup(Name))
    .Case("true", true)
    .Case("false", false)
    .Default(DefaultVal);
}

bool AnalyzerOptions::includeTemporaryDtorsInCFG() const {
  if (!IncludeTemporaryDtorsInCFG.hasValue())
    const_cast<llvm::Optional<bool> &>(IncludeTemporaryDtorsInCFG) =
      getBooleanOption("cfg-temporary-dtors", /*Default=*/false);
  
  return *IncludeTemporaryDtorsInCFG;
}

bool AnalyzerOptions::mayInlineCXXStandardLibrary() const {
  if (!InlineCXXStandardLibrary.hasValue())
    const_cast<llvm::Optional<bool> &>(InlineCXXStandardLibrary) =
      getBooleanOption("c++-stdlib-inlining", /*Default=*/true);
  
  return *InlineCXXStandardLibrary;
}

bool AnalyzerOptions::mayInlineTemplateFunctions() const {
  if (!InlineTemplateFunctions.hasValue())
    const_cast<llvm::Optional<bool> &>(InlineTemplateFunctions) =
      getBooleanOption("c++-template-inlining", /*Default=*/true);
  
  return *InlineTemplateFunctions;
}

bool AnalyzerOptions::mayInlineObjCMethod() const {
  if (!ObjCInliningMode.hasValue())
    const_cast<llvm::Optional<bool> &>(ObjCInliningMode) =
      getBooleanOption("objc-inlining", /*Default=*/true);

  return *ObjCInliningMode;
}

int AnalyzerOptions::getOptionAsInteger(StringRef Name, int DefaultVal) const {
  std::string OptStr = Config.lookup(Name);
  if (OptStr.empty())
    return DefaultVal;

  int Res = DefaultVal;
  assert(StringRef(OptStr).getAsInteger(10, Res) == false &&
         "analyzer-config option should be numeric.");

  return Res;
}

unsigned AnalyzerOptions::getAlwaysInlineSize() const {
  if (!AlwaysInlineSize.hasValue()) {
    unsigned DefaultSize = 3;
    const_cast<Optional<unsigned> &>(AlwaysInlineSize) =
      getOptionAsInteger("ipa-always-inline-size", DefaultSize);
  }

  return AlwaysInlineSize.getValue();
}
