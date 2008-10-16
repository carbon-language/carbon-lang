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
  
  /// ParseAST - Parse the entire file specified, notifying the ASTConsumer as
  /// the file is parsed.  This takes ownership of the ASTConsumer and
  /// ultimately deletes it.
  ///
  /// \param FreeMemory If false, the memory used for AST elements is
  /// not released.
  void ParseAST(Preprocessor &pp, ASTConsumer *C, 
                bool PrintStats = false, bool FreeMemory = true);

}  // end namespace clang

#endif
