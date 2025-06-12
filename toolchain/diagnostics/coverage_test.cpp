// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "absl/flags/flag.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/testing/coverage_helper.h"

ABSL_FLAG(std::string, testdata_manifest, "",
          "A path to a file containing repo-relative names of test files.");

namespace Carbon::Diagnostics {
namespace {

constexpr Kind Kinds[] = {
#define CARBON_DIAGNOSTIC_KIND(Name) Kind::Name,
#include "toolchain/diagnostics/diagnostic_kind.def"
};

constexpr Kind UntestedKinds[] = {
    // These exist only for unit tests.
    Kind::TestDiagnostic,
    Kind::TestDiagnosticNote,

    // Diagnosing erroneous install conditions, but test environments are
    // typically correct.
    Kind::CompilePreludeManifestError,
    Kind::DriverInstallInvalid,

    // These diagnose filesystem issues that are hard to unit test.
    Kind::ErrorReadingFile,
    Kind::ErrorStattingFile,
    Kind::FileTooLarge,

    // These aren't feasible to test with a normal testcase, but are tested in
    // lex/tokenized_buffer_test.cpp.
    Kind::TooManyTokens,
    Kind::UnsupportedCrLineEnding,
    Kind::UnsupportedLfCrLineEnding,

    // This is a little long but is tested in lex/numeric_literal_test.cpp.
    Kind::TooManyDigits,

    // TODO: This can only fire if the first message in a diagnostic is rooted
    // in a file other than the file being compiled. The language server
    // currently only supports compiling one file at a time. Do one of:
    // - When imports are supported, find a diagnostic whose first message isn't
    //   in the current file.
    // - Require all diagnostics produced by compiling have their first location
    //   be in the file being compiled, never an import.
    Kind::LanguageServerDiagnosticInWrongFile,
};

// Looks for diagnostic kinds that aren't covered by a file_test.
TEST(Coverage, Kind) {
  Testing::TestKindCoverage(absl::GetFlag(FLAGS_testdata_manifest),
                            R"(^ *// CHECK:STDERR: .* \[(\w+)\]$)",
                            llvm::ArrayRef(Kinds),
                            llvm::ArrayRef(UntestedKinds));
}

}  // namespace
}  // namespace Carbon::Diagnostics
