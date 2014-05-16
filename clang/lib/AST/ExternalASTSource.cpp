//===- ExternalASTSource.cpp - Abstract External AST Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides the default implementation of the ExternalASTSource 
//  interface, which enables construction of AST nodes from some external
//  source.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclarationName.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

ExternalASTSource::~ExternalASTSource() { }

void ExternalASTSource::PrintStats() { }

Decl *ExternalASTSource::GetExternalDecl(uint32_t ID) {
  return nullptr;
}

Selector ExternalASTSource::GetExternalSelector(uint32_t ID) {
  return Selector();
}

uint32_t ExternalASTSource::GetNumExternalSelectors() {
   return 0;
}

Stmt *ExternalASTSource::GetExternalDeclStmt(uint64_t Offset) {
  return nullptr;
}

CXXBaseSpecifier *
ExternalASTSource::GetExternalCXXBaseSpecifiers(uint64_t Offset) {
  return nullptr;
}

bool
ExternalASTSource::FindExternalVisibleDeclsByName(const DeclContext *DC,
                                                  DeclarationName Name) {
  return false;
}

void ExternalASTSource::completeVisibleDeclsMap(const DeclContext *DC) {
}

ExternalLoadResult
ExternalASTSource::FindExternalLexicalDecls(const DeclContext *DC,
                                            bool (*isKindWeWant)(Decl::Kind),
                                         SmallVectorImpl<Decl*> &Result) {
  return ELR_AlreadyLoaded;
}

void ExternalASTSource::getMemoryBufferSizes(MemoryBufferSizes &sizes) const { }

uint32_t ExternalASTSource::incrementGeneration(ASTContext &C) {
  uint32_t OldGeneration = CurrentGeneration;

  // Make sure the generation of the topmost external source for the context is
  // incremented. That might not be us.
  auto *P = C.getExternalSource();
  if (P && P != this)
    CurrentGeneration = P->incrementGeneration(C);
  else {
    // FIXME: Only bump the generation counter if the current generation number
    // has been observed?
    if (!++CurrentGeneration)
      llvm::report_fatal_error("generation counter overflowed", false);
  }

  return OldGeneration;
}
