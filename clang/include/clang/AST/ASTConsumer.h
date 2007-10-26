//===--- ASTConsumer.h - Abstract interface for reading ASTs ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTConsumer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTCONSUMER_H
#define LLVM_CLANG_AST_ASTCONSUMER_H

namespace clang {
  class ASTContext;
  
/// ASTConsumer - This is an abstract interface that should be implemented by
/// clients that read ASTs.  This abstraction layer allows the client to be
/// independent of the AST producer (e.g. parser vs AST dump file reader, etc).
class ASTConsumer {
public:
  virtual ~ASTConsumer();
  
  /// Initialize - This is called to initialize the consumer, providing the
  /// ASTContext and the file ID of the primary file.
  virtual void Initialize(ASTContext &Context, unsigned MainFileID) {
  }
  
  /// HandleTopLevelDecl - Handle the specified top-level declaration.
  ///
  virtual void HandleTopLevelDecl(Decl *D) {
  }
  
  /// HandleObjcMetaDataEmission - top level routine for objective-c
  /// metadata emission.
  virtual void HandleObjcMetaDataEmission() {
  }
  
  /// PrintStats - If desired, print any statistics.
  virtual void PrintStats() {
  }
};

} // end namespace clang.

#endif
