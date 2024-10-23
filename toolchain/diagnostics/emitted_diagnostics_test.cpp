// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "absl/flags/flag.h"
#include "common/set.h"
#include "llvm/ADT/StringExtras.h"
#include "re2/re2.h"
#include "toolchain/diagnostics/diagnostic_kind.h"

ABSL_FLAG(std::string, testdata_manifest, "",
          "A path to a file containing repo-relative names of test files.");

namespace Carbon {
namespace {

constexpr DiagnosticKind Diagnostics[] = {
#define CARBON_DIAGNOSTIC_KIND(Name) DiagnosticKind::Name,
#include "toolchain/diagnostics/diagnostic_kind.def"
};

// Returns true for diagnostics which have no tests. In general, diagnostics
// should be tested.
static auto IsUntestedDiagnostic(DiagnosticKind diagnostic_kind) -> bool {
  switch (diagnostic_kind) {
    case DiagnosticKind::TestDiagnostic:
    case DiagnosticKind::TestDiagnosticNote:
      // These exist only for unit tests.
      return true;
    case DiagnosticKind::ErrorReadingFile:
    case DiagnosticKind::ErrorStattingFile:
    case DiagnosticKind::FileTooLarge:
      // These diagnose filesystem issues that are hard to unit test.
      return true;
    case DiagnosticKind::ArrayBoundTooLarge:
      // Int literals are currently limited to i32. Once that's fixed, this
      // should be tested.
      return true;
    case DiagnosticKind::ExternLibraryInImporter:
    case DiagnosticKind::ExternLibraryOnDefinition:
    case DiagnosticKind::HexadecimalEscapeMissingDigits:
    case DiagnosticKind::ImplOfUndefinedInterface:
    case DiagnosticKind::IncompleteTypeInFunctionParam:
    case DiagnosticKind::InvalidDigit:
    case DiagnosticKind::InvalidDigitSeparator:
    case DiagnosticKind::InvalidHorizontalWhitespaceInString:
    case DiagnosticKind::MismatchedIndentInString:
    case DiagnosticKind::ModifierPrivateNotAllowed:
    case DiagnosticKind::MultiLineStringWithDoubleQuotes:
    case DiagnosticKind::NameAmbiguousDueToExtend:
    case DiagnosticKind::TooManyDigits:
    case DiagnosticKind::UnaryOperatorRequiresWhitespace:
    case DiagnosticKind::UnicodeEscapeSurrogate:
    case DiagnosticKind::UnicodeEscapeTooLarge:
    case DiagnosticKind::UnknownBaseSpecifier:
    case DiagnosticKind::UnsupportedCRLineEnding:
    case DiagnosticKind::UnsupportedLFCRLineEnding:
      // TODO: Should look closer at these, but adding tests is a high risk of
      // loss in merge conflicts due to the amount of tests being changed right
      // now.
      return true;
    case DiagnosticKind::TooManyTokens:
      // This isn't feasible to test with a normal testcase, but is tested in
      // lex/tokenized_buffer_test.cpp.
      return true;
    default:
      return false;
  }
}

TEST(EmittedDiagnostics, Verify) {
  std::ifstream manifest_in(absl::GetFlag(FLAGS_testdata_manifest));
  ASSERT_TRUE(manifest_in.good());

  RE2 diagnostic_re(R"(\w\((\w+)\): )");
  ASSERT_TRUE(diagnostic_re.ok()) << diagnostic_re.error();

  Set<std::string> emitted_diagnostics;

  std::string test_filename;
  while (std::getline(manifest_in, test_filename)) {
    std::ifstream test_in(test_filename);
    ASSERT_TRUE(test_in.good());

    std::string line;
    while (std::getline(test_in, line)) {
      std::string diagnostic;
      if (RE2::PartialMatch(line, diagnostic_re, &diagnostic)) {
        emitted_diagnostics.Insert(diagnostic);
      }
    }
  }

  llvm::SmallVector<llvm::StringRef> missing_diagnostics;
  for (auto diagnostic_kind : Diagnostics) {
    if (IsUntestedDiagnostic(diagnostic_kind)) {
      EXPECT_FALSE(emitted_diagnostics.Erase(diagnostic_kind.name()))
          << diagnostic_kind
          << " was previously untested, and is now tested. That's good, but "
             "please remove it from IsUntestedDiagnostic.";
      continue;
    }
    if (!emitted_diagnostics.Erase(diagnostic_kind.name())) {
      missing_diagnostics.push_back(diagnostic_kind.name());
    }
  }

  constexpr llvm::StringLiteral Bullet = "\n  - ";

  std::sort(missing_diagnostics.begin(), missing_diagnostics.end());
  EXPECT_TRUE(missing_diagnostics.empty())
      << "Some diagnostics have no tests:" << Bullet
      << llvm::join(missing_diagnostics, Bullet);

  llvm::SmallVector<std::string> unexpected_matches;
  emitted_diagnostics.ForEach(
      [&](const std::string& match) { unexpected_matches.push_back(match); });
  std::sort(unexpected_matches.begin(), unexpected_matches.end());
  EXPECT_TRUE(unexpected_matches.empty())
      << "Matched things that don't appear to be diagnostics:" << Bullet
      << llvm::join(unexpected_matches, Bullet);
}

}  // namespace
}  // namespace Carbon
