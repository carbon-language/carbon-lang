//===- ChainedIncludesSource.h - Chained PCHs in Memory ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ChainedIncludesSource class, which converts headers
//  to chained PCHs in memory, mainly used for testing.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SERIALIZATION_CHAINEDINCLUDESSOURCE_H
#define LLVM_CLANG_SERIALIZATION_CHAINEDINCLUDESSOURCE_H

#include "clang/Sema/ExternalSemaSource.h"
#include <vector>

namespace clang {
  class CompilerInstance;

class ChainedIncludesSource : public ExternalSemaSource {
public:
  virtual ~ChainedIncludesSource();

  static IntrusiveRefCntPtr<ChainedIncludesSource> create(CompilerInstance &CI);

  ExternalSemaSource &getFinalReader() const { return *FinalReader; }

private:
  std::vector<CompilerInstance *> CIs;
  IntrusiveRefCntPtr<ExternalSemaSource> FinalReader;

  
protected:

//===----------------------------------------------------------------------===//
// ExternalASTSource interface.
//===----------------------------------------------------------------------===//

  Decl *GetExternalDecl(uint32_t ID) override;
  Selector GetExternalSelector(uint32_t ID) override;
  uint32_t GetNumExternalSelectors() override;
  Stmt *GetExternalDeclStmt(uint64_t Offset) override;
  CXXBaseSpecifier *GetExternalCXXBaseSpecifiers(uint64_t Offset) override;
  bool FindExternalVisibleDeclsByName(const DeclContext *DC,
                                      DeclarationName Name) override;
  ExternalLoadResult FindExternalLexicalDecls(const DeclContext *DC,
                                bool (*isKindWeWant)(Decl::Kind),
                                SmallVectorImpl<Decl*> &Result) override;
  void CompleteType(TagDecl *Tag) override;
  void CompleteType(ObjCInterfaceDecl *Class) override;
  void StartedDeserializing() override;
  void FinishedDeserializing() override;
  void StartTranslationUnit(ASTConsumer *Consumer) override;
  void PrintStats() override;

  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  void getMemoryBufferSizes(MemoryBufferSizes &sizes) const override;

//===----------------------------------------------------------------------===//
// ExternalSemaSource interface.
//===----------------------------------------------------------------------===//

  void InitializeSema(Sema &S) override;
  void ForgetSema() override;
  void ReadMethodPool(Selector Sel) override;
  bool LookupUnqualified(LookupResult &R, Scope *S) override;
};

}

#endif
