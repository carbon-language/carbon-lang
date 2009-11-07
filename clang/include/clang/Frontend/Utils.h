//===--- Utils.h - Misc utilities for the front-end------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header contains miscellaneous utilities for various front-end actions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_UTILS_H
#define LLVM_CLANG_FRONTEND_UTILS_H

#include <vector>
#include <string>

namespace llvm {
class raw_ostream;
class raw_fd_ostream;
}

namespace clang {
class ASTConsumer;
class Decl;
class Diagnostic;
class IdentifierTable;
class LangOptions;
class MinimalAction;
class Preprocessor;
class PreprocessorOptions;
class SourceManager;
class Stmt;
class TargetInfo;

/// InitializePreprocessor - Initialize the preprocessor getting it and the
/// environment ready to process a single file.
///
void InitializePreprocessor(Preprocessor &PP,
                            const PreprocessorOptions &PPOpts);

/// ProcessWarningOptions - Initialize the diagnostic client and process the
/// warning options specified on the command line.
bool ProcessWarningOptions(Diagnostic &Diags,
                           std::vector<std::string> &Warnings,
                           bool Pedantic, bool PedanticErrors,
                           bool NoWarnings);

/// DoPrintPreprocessedInput - Implement -E -dM mode.
void DoPrintMacros(Preprocessor &PP, llvm::raw_ostream* OS);

/// DoPrintPreprocessedInput - Implement -E mode.
void DoPrintPreprocessedInput(Preprocessor &PP, llvm::raw_ostream* OS,
                              bool EnableCommentOutput,
                              bool EnableMacroCommentOutput,
                              bool DisableLineMarkers,
                              bool DumpDefines);

/// RewriteMacrosInInput - Implement -rewrite-macros mode.
void RewriteMacrosInInput(Preprocessor &PP, llvm::raw_ostream* OS);

/// RewriteMacrosInInput - A simple test for the TokenRewriter class.
void DoRewriteTest(Preprocessor &PP, llvm::raw_ostream* OS);

/// CreatePrintParserActionsAction - Return the actions implementation that
/// implements the -parse-print-callbacks option.
MinimalAction *CreatePrintParserActionsAction(Preprocessor &PP,
                                              llvm::raw_ostream* OS);

/// CheckDiagnostics - Gather the expected diagnostics and check them.
bool CheckDiagnostics(Preprocessor &PP);

/// AttachDependencyFileGen - Create a dependency file generator, and attach
/// it to the given preprocessor.  This takes ownership of the output stream.
void AttachDependencyFileGen(Preprocessor *PP, llvm::raw_ostream *OS,
                             std::vector<std::string> &Targets,
                             bool IncludeSystemHeaders, bool PhonyTarget);

/// CacheTokens - Cache tokens for use with PCH. Note that this requires
/// a seekable stream.
void CacheTokens(Preprocessor& PP, llvm::raw_fd_ostream* OS);

}  // end namespace clang

#endif
