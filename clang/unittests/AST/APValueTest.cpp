//===- unittests/AST/APValueTest.cpp - APValue tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/APValue.h"

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class DiagnosticOutputGetter {
  class LastDiagnosticString : public DiagnosticConsumer {
    SmallString<64> LastDiagnostic;
  public:
    virtual void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                  const Diagnostic &Info) {
      LastDiagnostic.clear();
      Info.FormatDiagnostic(LastDiagnostic);
    }

    StringRef get() const { return LastDiagnostic; }

    virtual DiagnosticConsumer *clone(DiagnosticsEngine &Diags) const {
      return new LastDiagnosticString();
    }
  };

  const IntrusiveRefCntPtr<DiagnosticIDs> DiagIDs;
  const unsigned diag_just_format;
  LastDiagnosticString LastDiagnostic;
  DiagnosticsEngine Diag;

public:
  DiagnosticOutputGetter()
    : DiagIDs(new DiagnosticIDs),
      diag_just_format(DiagIDs->getCustomDiagID(DiagnosticIDs::Error, "%0")),
      Diag(DiagIDs, &LastDiagnostic, false) {
  }

  template<typename T>
  std::string operator()(const T& value) {
    Diag.Report(diag_just_format) << value;
    return LastDiagnostic.get().str();
  }
};

TEST(APValue, Diagnostics) {
  DiagnosticOutputGetter GetDiagnosticOutput;

  EXPECT_EQ("Uninitialized", GetDiagnosticOutput(APValue()));
  EXPECT_EQ("5", GetDiagnosticOutput(APValue(APSInt(APInt(16, 5)))));
  EXPECT_EQ("3.141590e+00",
            GetDiagnosticOutput(APValue(APFloat(APFloat::IEEEdouble,
                                                "3.14159"))));
  EXPECT_EQ("3+4i",
            GetDiagnosticOutput(APValue(APSInt(APInt(16, 3)),
                                        APSInt(APInt(16, 4)))));
  EXPECT_EQ("3.200000e+00+5.700000e+00i",
            GetDiagnosticOutput(APValue(
                                  APFloat(APFloat::IEEEdouble, "3.2"),
                                  APFloat(APFloat::IEEEdouble, "5.7"))));
  APValue V[] = {
    APValue(APSInt(APInt(16, 3))),
    APValue(APSInt(APInt(16, 4))),
    APValue(APSInt(APInt(16, 5)))
  };
  EXPECT_EQ("[3, 4, 5]",
            GetDiagnosticOutput(APValue(V, array_lengthof(V))));
}

} // anonymous namespace
