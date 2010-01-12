//===- CIndexUSR.cpp - Clang-C Source Indexing Library --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the generation and use of USRs from CXEntities.
//
//===----------------------------------------------------------------------===//

#include "CIndexer.h"

extern "C" {

// Some notes on CXEntity:
//
// - Since the 'ordinary' namespace includes functions, data, typedefs,
// ObjC interfaces, thecurrent algorithm is a bit naive (resulting in one
// entity for 2 different types). For example:
//
// module1.m: @interface Foo @end Foo *x;
// module2.m: void Foo(int);
//
// - Since the unique name spans translation units, static data/functions
// within a CXTranslationUnit are *not* currently represented by entities.
// As a result, there will be no entity for the following:
//
// module.m: static void Foo() { }
//
  
static inline Entity GetEntity(const CXEntity &E) {
  return Entity::getFromOpaquePtr(E.data);
}
  
static inline ASTUnit *GetTranslationUnit(CXTranslationUnit TU) {
  return (ASTUnit*) TU;
}

static inline ASTContext &GetASTContext(CXTranslationUnit TU) {
  return GetTranslationUnit(TU)->getASTContext();
}

static inline CXEntity NullCXEntity() {
  CXEntity CE;
  CE.index = NULL;
  CE.data = NULL;
  return CE;
}
  
static inline CXEntity MakeEntity(CXIndex CIdx, const Entity &E) {
  CXEntity CE;
  CE.index = CIdx;
  CE.data = E.getAsOpaquePtr();
  return CE;
}

static inline Program &GetProgram(CXIndex CIdx) {
  return ((CIndexer*) CIdx)->getProgram();
}
 
/// clang_getDeclaration() maps from a CXEntity to the matching CXDecl (if any)
///  in a specified translation unit.
CXDecl clang_getDeclaration(CXEntity CE, CXTranslationUnit TU) {
  return (CXDecl) GetEntity(CE).getDecl(GetASTContext(TU));
}

  
CXEntity clang_getEntityFromDecl(CXIndex CIdx, CXDecl CE) {
  if (Decl *D = (Decl *) CE)
    return MakeEntity(CIdx, Entity::get(D, GetProgram(CIdx)));
  return NullCXEntity();
}
  
CXString clang_getUSR(CXEntity) {
  return CIndexer::createCXString("");
}

} // end extern "C"
