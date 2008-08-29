//===--- ASTConsumer.h - Abstract interface for reading ASTs ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  class TranslationUnit;
  class Decl;
  class TagDecl;
  class HandleTagDeclDefinition;
  
/// ASTConsumer - This is an abstract interface that should be implemented by
/// clients that read ASTs.  This abstraction layer allows the client to be
/// independent of the AST producer (e.g. parser vs AST dump file reader, etc).
class ASTConsumer {
public:
  virtual ~ASTConsumer();
  
  /// Initialize - This is called to initialize the consumer, providing the
  /// ASTContext.
  virtual void Initialize(ASTContext &Context) {}
  
  virtual void InitializeTU(TranslationUnit& TU);
  
  /// HandleTopLevelDecl - Handle the specified top-level declaration.  This is
  ///  called by the parser to process every top-level Decl*. Note that D can
  ///  be the head of a chain of Decls (e.g. for `int a, b` the chain will have
  ///  two elements). Use ScopedDecl::getNextDeclarator() to walk the chain.
  virtual void HandleTopLevelDecl(Decl *D) {}
  
  /// HandleTranslationUnit - This method is called when the ASTs for entire
  ///  translation unit have been parsed.
  virtual void HandleTranslationUnit(TranslationUnit& TU) {}    
  
  /// HandleTagDeclDefinition - This callback is invoked each time a TagDecl
  /// (e.g. struct, union, enum, class) is completed.  This allows the client to
  /// hack on the type, which can occur at any point in the file (because these
  /// can be defined in declspecs).
  virtual void HandleTagDeclDefinition(TagDecl *D) {}
  
  /// PrintStats - If desired, print any statistics.
  virtual void PrintStats() {
  }
};

} // end namespace clang.

#endif
