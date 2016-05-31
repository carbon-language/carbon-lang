//===-- IncludeFixer.h - Include inserter -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXER_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXER_H

#include "IncludeFixerContext.h"
#include "SymbolIndexManager.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include <memory>
#include <vector>

namespace clang {

class CompilerInvocation;
class DiagnosticConsumer;
class FileManager;
class PCHContainerOperations;

namespace include_fixer {

class IncludeFixerActionFactory : public clang::tooling::ToolAction {
public:
  /// \param SymbolIndexMgr A source for matching symbols to header files.
  /// \param Context A context for the symbol being queried.
  /// \param StyleName Fallback style for reformatting.
  /// \param MinimizeIncludePaths whether inserted include paths are optimized.
  IncludeFixerActionFactory(SymbolIndexManager &SymbolIndexMgr,
                            IncludeFixerContext &Context, StringRef StyleName,
                            bool MinimizeIncludePaths = true);

  ~IncludeFixerActionFactory() override;

  bool
  runInvocation(clang::CompilerInvocation *Invocation,
                clang::FileManager *Files,
                std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps,
                clang::DiagnosticConsumer *Diagnostics) override;

private:
  /// The client to use to find cross-references.
  SymbolIndexManager &SymbolIndexMgr;

  /// The context that contains all information about the symbol being queried.
  IncludeFixerContext &Context;

  /// Whether inserted include paths should be optimized.
  bool MinimizeIncludePaths;

  /// The fallback format style for formatting after insertion if no
  /// clang-format config file was found.
  std::string FallbackStyle;
};

/// Create replacements for the header being inserted. The replacements will
/// insert a header before the first #include in \p Code, and sort all headers
/// with the given clang-format style.
///
/// \param Code The source code.
/// \param FilePath The source file path.
/// \param Header The header being inserted.
/// \param FirstIncludeOffset The insertion point for new include directives.
/// The default value -1U means inserting the header at the first line, and if
/// there is no #include block, it will create a header block by inserting a
/// newline.
/// \param Style clang-format style being used.
///
/// \return Replacements for inserting and sorting headers.
tooling::Replacements createInsertHeaderReplacements(
    StringRef Code, StringRef FilePath, StringRef Header,
    unsigned FirstIncludeOffset = -1U,
    const clang::format::FormatStyle &Style = clang::format::getLLVMStyle());

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXER_H
