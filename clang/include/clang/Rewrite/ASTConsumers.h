//===--- ASTConsumers.h - ASTConsumer implementations -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AST Consumers.
//
//===----------------------------------------------------------------------===//

#ifndef REWRITE_ASTCONSUMERS_H
#define REWRITE_ASTCONSUMERS_H

#include <string>

namespace llvm {
  class raw_ostream;
}
namespace clang {

class ASTConsumer;
class Diagnostic;
class LangOptions;
class Preprocessor;

// ObjC rewriter: attempts to rewrite ObjC constructs into pure C code.
// This is considered experimental, and only works with Apple's ObjC runtime.
ASTConsumer *CreateObjCRewriter(const std::string &InFile,
                                llvm::raw_ostream *OS,
                                Diagnostic &Diags,
                                const LangOptions &LOpts,
                                bool SilenceRewriteMacroWarning);

/// CreateHTMLPrinter - Create an AST consumer which rewrites source code to
/// HTML with syntax highlighting suitable for viewing in a web-browser.
ASTConsumer *CreateHTMLPrinter(llvm::raw_ostream *OS, Preprocessor &PP,
                               bool SyntaxHighlight = true,
                               bool HighlightMacros = true);

} // end clang namespace

#endif
