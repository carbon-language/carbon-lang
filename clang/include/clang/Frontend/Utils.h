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
class Preprocessor;
class MinimalAction;
class TargetInfo;
class Diagnostic;
class ASTConsumer;
class IdentifierTable;
class SourceManager;
class PreprocessorFactory;
class LangOptions;
class Decl;
class Stmt;
class ASTContext;
class SourceLocation;

namespace idx {
class ASTLocation;
}

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

/// \brief Returns the AST node that a source location points to.
///
/// Returns a pair of Decl* and Stmt*. If no AST node is found for the source
/// location, the pair will contain null pointers.
///
/// If the source location points to just a declaration, the statement part of
/// the pair will be null, e.g.,
/// @code
///   int foo;
/// @endcode
/// If the source location points at 'foo', the pair will contain the VarDecl
/// of foo and a null Stmt.
///
/// If the source location points to a statement node, the returned declaration
/// will be the immediate 'parent' declaration of the statement node, e.g.,
/// @code
///   void f() {
///     int foo = 100;
///     ++foo;
///   }
/// @endcode
/// Pointing at '100' will return a <VarDecl 'foo', IntegerLiteral '100'> pair.
/// Pointing at '++foo' will return a <FunctionDecl 'f', UnaryOperator> pair.
///
idx::ASTLocation ResolveLocationInAST(ASTContext &Ctx, SourceLocation Loc);

}  // end namespace clang

#endif
