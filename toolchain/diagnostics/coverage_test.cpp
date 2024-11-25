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

    // These aren't feasible to test with a normal testcase, but are tested in
    // lex/tokenized_buffer_test.cpp.
    DiagnosticKind::TooManyTokens,
    DiagnosticKind::UnsupportedCRLineEnding,
    DiagnosticKind::UnsupportedLFCRLineEnding,

    // This is a little long but is tested in lex/numeric_literal_test.cpp.
    DiagnosticKind::TooManyDigits,
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
