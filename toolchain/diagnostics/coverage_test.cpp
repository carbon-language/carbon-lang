// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "absl/flags/flag.h"
#include "toolchain/diagnostics/kind.h"
#include "toolchain/testing/coverage_helper.h"

ABSL_FLAG(std::string, testdata_manifest, "",
          "A path to a file containing repo-relative names of test files.");

namespace Carbon::Diagnostics {
namespace {

constexpr Kind Kinds[] = {
#define CARBON_DIAGNOSTIC_KIND(Name) Kind::Name,
#include "toolchain/diagnostics/kind.def"
};

constexpr Kind UntestedKinds[] = {
    // These exist only for unit tests.
    Kind::TestDiagnostic,
    Kind::TestDiagnosticOnScope,

    // Diagnosing erroneous install conditions, but test environments are
    // typically correct.
    Kind::BuildTempDirectoryCreationError,
    Kind::BuildTempDirectoryDeletionError,
    Kind::CompileCoreManifestError,
    Kind::CompilePreludeManifestError,
    Kind::ConfigFailedToReadDigest,
    Kind::ConfigFailedToSetupTarget,
    Kind::DriverInstallInvalid,
    Kind::LinkCarbonPreludeBuildFailed,

    // These diagnose filesystem issues that are hard to unit test.
    Kind::ErrorReadingFile,
    Kind::ErrorStattingFile,
    Kind::FileTooLarge,
    Kind::FailureBuildingRuntimes,
    Kind::FailureRunningClang,
    Kind::FailureRunningClangToLink,

    // These aren't feasible to test with a normal testcase, but are tested in
    // lex/tokenized_buffer_test.cpp.
    Kind::TooManyTokens,
    Kind::UnsupportedCrLineEnding,
    Kind::UnsupportedLfCrLineEnding,

    // This is a little long but is tested in lex/numeric_literal_test.cpp.
    Kind::TooManyDigits,

    // Producing an emit failure may be infeasible.
    Kind::CodeGenUnableToEmit,

    // Degradation for a Clang note that leads a flushed buffer instead of
    // trailing the diagnostic it explains, which needs a mistimed flush or a
    // Clang bug to produce.
    Kind::CppInteropStrayNote,

    // TODO: This can only fire if a diagnostic's message is rooted in a file
    // other than the file being compiled. The language server currently only
    // supports compiling one file at a time. Do one of:
    // - When imports are supported, find a diagnostic whose message isn't in
    //   the current file.
    // - Require all diagnostics produced by compiling have their message's
    //   location be in the file being compiled, never an import.
    Kind::LanguageServerDiagnosticInWrongFile,
};

// Looks for diagnostic kinds that aren't covered by a file_test.
//
// A line names either a kind or an attached label, and there is no list of
// labels to enumerate from here, so a match that isn't a kind is left alone.
// `check_diagnostics.py` is what covers them: it reads the declarations and the
// testdata directly, so it checks that every label is exercised and that every
// name a test matches on is a kind or a label that exists.
TEST(Coverage, Kind) {
  Testing::TestKindCoverage(
      absl::GetFlag(FLAGS_testdata_manifest),
      R"(^ *// CHECK:STDERR: .* \[(\w+)\]$)", llvm::ArrayRef(Kinds),
      llvm::ArrayRef(UntestedKinds), /*allow_unlisted_matches=*/true);
}

}  // namespace
}  // namespace Carbon::Diagnostics
