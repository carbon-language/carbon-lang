//===--- ParseAST.h - Define the ParseAST method ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the clang::ParseAST method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSEAST_H
#define LLVM_CLANG_PARSE_PARSEAST_H

namespace clang {
  class Preprocessor;
  class ASTConsumer;
  class ASTContext;
  class CodeCompleteConsumer;
  class Sema;

  /// \brief Parse the entire file specified, notifying the ASTConsumer as
  /// the file is parsed.
  ///
  /// This operation inserts the parsed decls into the translation
  /// unit held by Ctx.
  ///
  /// \param CompleteTranslationUnit When true, the parsed file is
  /// considered to be a complete translation unit, and any
  /// end-of-translation-unit wrapup will be performed.
  ///
  /// \param CompletionConsumer If given, an object to consume code completion
  /// results.
  void ParseAST(Preprocessor &pp, ASTConsumer *C,
                ASTContext &Ctx, bool PrintStats = false,
                bool CompleteTranslationUnit = true,
                CodeCompleteConsumer *CompletionConsumer = 0);

  /// \brief Parse the main file known to the preprocessor, producing an 
  /// abstract syntax tree.
  void ParseAST(Sema &S, bool PrintStats = false);
  
}  // end namespace clang

#endif
