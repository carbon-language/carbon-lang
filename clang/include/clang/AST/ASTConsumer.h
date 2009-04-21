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
  class DeclGroupRef;
  class TagDecl;
  class HandleTagDeclDefinition;
  class SemaConsumer; // layering violation required for safe SemaConsumer
  class VarDecl;

/// ASTConsumer - This is an abstract interface that should be implemented by
/// clients that read ASTs.  This abstraction layer allows the client to be
/// independent of the AST producer (e.g. parser vs AST dump file reader, etc).
class ASTConsumer {
  /// \brief Whether this AST consumer also requires information about
  /// semantic analysis.
  bool SemaConsumer;

  friend class SemaConsumer;

public:
  ASTConsumer() : SemaConsumer(false) { }

  virtual ~ASTConsumer() {}
  
  /// Initialize - This is called to initialize the consumer, providing the
  /// ASTContext and the Action.
  virtual void Initialize(ASTContext &Context) {}
  
  /// HandleTopLevelDecl - Handle the specified top-level declaration.  This is
  ///  called by the parser to process every top-level Decl*. Note that D can
  ///  be the head of a chain of Decls (e.g. for `int a, b` the chain will have
  ///  two elements). Use Decl::getNextDeclarator() to walk the chain.
  virtual void HandleTopLevelDecl(DeclGroupRef D);
  
  /// HandleTranslationUnit - This method is called when the ASTs for entire
  ///  translation unit have been parsed.
  virtual void HandleTranslationUnit(ASTContext &Ctx) {}    
  
  /// HandleTagDeclDefinition - This callback is invoked each time a TagDecl
  /// (e.g. struct, union, enum, class) is completed.  This allows the client to
  /// hack on the type, which can occur at any point in the file (because these
  /// can be defined in declspecs).
  virtual void HandleTagDeclDefinition(TagDecl *D) {}
  
  /// \brief Callback invoked at the end of a translation unit to
  /// notify the consumer that the given tentative definition should
  /// be completed.
  ///
  /// The variable declaration itself will be a tentative
  /// definition. If it had an incomplete array type, its type will
  /// have already been changed to an array of size 1. However, the
  /// declaration remains a tentative definition and has not been
  /// modified by the introduction of an implicit zero initializer.
  virtual void CompleteTentativeDefinition(VarDecl *D) {}

  /// PrintStats - If desired, print any statistics.
  virtual void PrintStats() {
  }

  // Support isa/cast/dyn_cast
  static bool classof(const ASTConsumer *) { return true; }
};

} // end namespace clang.

#endif
