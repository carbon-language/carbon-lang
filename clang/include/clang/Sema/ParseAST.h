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

#ifndef LLVM_CLANG_SEMA_PARSEAST_H
#define LLVM_CLANG_SEMA_PARSEAST_H

namespace clang {
  class Preprocessor;
  class ASTConsumer;
  class TranslationUnit;

  /// ParseAST - Parse the entire file specified, notifying the ASTConsumer as
  /// the file is parsed.  This inserts the parsed decls into TU.
  ///
  void ParseAST(Preprocessor &pp, ASTConsumer *C, 
                TranslationUnit &TU, bool PrintStats = false);

}  // end namespace clang

#endif
