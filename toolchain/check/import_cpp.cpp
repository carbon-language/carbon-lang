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

// Generates C++ file contents to #include all requested imports.
static auto GenerateCppIncludesHeaderCode(
    llvm::ArrayRef<std::pair<llvm::StringRef, SemIRLoc>> imports)
    -> std::string {
  std::string code;
  llvm::raw_string_ostream code_stream(code);
  for (const auto& [path, _] : imports) {
    code_stream << "#include \"" << FormatEscaped(path) << "\"\n";
  }
  return code;
}

auto ImportCppFiles(
    Context& context, llvm::StringRef importing_file_path,
    llvm::ArrayRef<std::pair<llvm::StringRef, SemIRLoc>> imports,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs) -> void {
  size_t num_imports = imports.size();
  if (num_imports == 0) {
    return;
  }

  // TODO: Use all import locations by referring each Clang diagnostic to the
  // relevant import.
  SemIRLoc loc = imports.back().second;

  std::string diagnostics_str;
  llvm::raw_string_ostream diagnostics_stream(diagnostics_str);

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options(
      new clang::DiagnosticOptions());
  clang::TextDiagnosticPrinter diagnostics_consumer(diagnostics_stream,
                                                    diagnostic_options.get());
  // TODO: Share compilation flags with ClangRunner.
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      GenerateCppIncludesHeaderCode(imports), {},
      (importing_file_path + ".generated.cpp_imports.h").str(), "clang-tool",
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
        "{0} error{0:s} and {1} warning{1:s} in {2} `Cpp` import{2:s}:\n{3}",
        IntAsSelect, IntAsSelect, IntAsSelect, std::string);
    context.emitter().Emit(loc, CppInteropParseError, num_errors, num_warnings,
                           num_imports, diagnostics_str);
  } else if (num_warnings > 0) {
    CARBON_DIAGNOSTIC(CppInteropParseWarning, Warning,
                      "{0} warning{0:s} in `Cpp` {1} import{1:s}:\n{2}",
                      IntAsSelect, IntAsSelect, std::string);
    context.emitter().Emit(loc, CppInteropParseWarning, num_warnings,
                           num_imports, diagnostics_str);
  }
}

}  // namespace Carbon::Check
