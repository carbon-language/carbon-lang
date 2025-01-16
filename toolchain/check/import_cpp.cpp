// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/import_cpp.h"

#include <memory>
#include <string>

#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/format_providers.h"

namespace Carbon::Check {

auto ImportCppFile(Context& context, SemIRLoc loc,
                   llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                   llvm::StringRef file_path, llvm::StringRef code) -> void {
  RawStringOstream diagnostics_stream;

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options(
      new clang::DiagnosticOptions());
  clang::TextDiagnosticPrinter diagnostics_consumer(diagnostics_stream,
                                                    diagnostic_options.get());
  // TODO: Share compilation flags with ClangRunner.
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      code, {}, file_path, "clang-tool",
      std::make_shared<clang::PCHContainerOperations>(),
      clang::tooling::getClangStripDependencyFileAdjuster(),
      clang::tooling::FileContentMappings(), &diagnostics_consumer, fs);
  // TODO: Implement and use a DynamicRecursiveASTVisitor to traverse the AST.
  int num_errors = diagnostics_consumer.getNumErrors();
  int num_warnings = diagnostics_consumer.getNumWarnings();
  if (num_errors > 0) {
    // TODO: Remove the warnings part when there are no warnings.
    CARBON_DIAGNOSTIC(
        CppInteropParseError, Error,
        "{0} error{0:s} and {1} warning{1:s} in `Cpp` import `{2}`:\n{3}",
        IntAsSelect, IntAsSelect, std::string, std::string);
    context.emitter().Emit(loc, CppInteropParseError, num_errors, num_warnings,
                           file_path.str(), diagnostics_stream.TakeStr());
  } else if (num_warnings > 0) {
    CARBON_DIAGNOSTIC(CppInteropParseWarning, Warning,
                      "{0} warning{0:s} in `Cpp` import `{1}`:\n{2}",
                      IntAsSelect, std::string, std::string);
    context.emitter().Emit(loc, CppInteropParseWarning, num_warnings,
                           file_path.str(), diagnostics_stream.TakeStr());
  }
}

}  // namespace Carbon::Check
