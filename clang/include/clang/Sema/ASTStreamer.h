//===--- ASTStreamer.h - Stream ASTs for top-level decls --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTStreamer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTSTREAMER_H
#define LLVM_CLANG_AST_ASTSTREAMER_H

namespace clang {
  class Preprocessor;
  class ASTContext;
  class Decl;
  class ASTConsumer;
  
  /// ParseAST - Parse the entire file specified, notifying the ASTConsumer as
  /// the file is parsed.  This takes ownership of the ASTConsumer and
  /// ultimately deletes it.
  void ParseAST(Preprocessor &pp, ASTConsumer *C, bool PrintStats = false);
}  // end namespace clang

#endif
