// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "absl/flags/flag.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/testing/coverage_helper.h"

ABSL_FLAG(std::string, testdata_manifest, "",
          "A path to a file containing repo-relative names of test files.");

namespace Carbon {
namespace {

constexpr DiagnosticKind DiagnosticKinds[] = {
#define CARBON_DIAGNOSTIC_KIND(Name) DiagnosticKind::Name,
#include "toolchain/diagnostics/diagnostic_kind.def"
};

constexpr DiagnosticKind UntestedDiagnosticKinds[] = {
    // These exist only for unit tests.
    DiagnosticKind::TestDiagnostic,
    DiagnosticKind::TestDiagnosticNote,

    // These diagnose filesystem issues that are hard to unit test.
    DiagnosticKind::ErrorReadingFile,
    DiagnosticKind::ErrorStattingFile,
    DiagnosticKind::FileTooLarge,

    // Int literals are currently limited to i32. Once that's fixed, this
    // should be tested.
    DiagnosticKind::ArrayBoundTooLarge,

    // This isn't feasible to test with a normal testcase, but is tested in
    // lex/tokenized_buffer_test.cpp.
    DiagnosticKind::TooManyTokens,

    // TODO: Should look closer at these, but adding tests is a high risk of
    // loss in merge conflicts due to the amount of tests being changed right
    // now.
    DiagnosticKind::ExternLibraryInImporter,
    DiagnosticKind::ExternLibraryOnDefinition,
    DiagnosticKind::HexadecimalEscapeMissingDigits,
    DiagnosticKind::IncompleteTypeInFunctionParam,
    DiagnosticKind::InvalidDigit,
    DiagnosticKind::InvalidDigitSeparator,
    DiagnosticKind::InvalidHorizontalWhitespaceInString,
    DiagnosticKind::MismatchedIndentInString,
    DiagnosticKind::ModifierPrivateNotAllowed,
    DiagnosticKind::MultiLineStringWithDoubleQuotes,
    DiagnosticKind::TooManyDigits,
    DiagnosticKind::UnaryOperatorRequiresWhitespace,
    DiagnosticKind::UnicodeEscapeSurrogate,
    DiagnosticKind::UnicodeEscapeTooLarge,
    DiagnosticKind::UnknownBaseSpecifier,
    DiagnosticKind::UnsupportedCRLineEnding,
    DiagnosticKind::UnsupportedLFCRLineEnding,
};

// Looks for diagnostic kinds that aren't covered by a file_test.
TEST(Coverage, DiagnosticKind) {
  Testing::TestKindCoverage(absl::GetFlag(FLAGS_testdata_manifest),
                            R"(^ *// CHECK:STDERR: .*\.carbon:.* \[(\w+)\]$)",
                            llvm::ArrayRef(DiagnosticKinds),
                            llvm::ArrayRef(UntestedDiagnosticKinds));
}

}  // namespace
}  // namespace Carbon
