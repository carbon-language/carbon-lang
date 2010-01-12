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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h";
#include "clang/AST/DeclVisitor.h";

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
  
//===----------------------------------------------------------------------===//
// USR generation.
//===----------------------------------------------------------------------===//

namespace {
class USRGenerator : public DeclVisitor<USRGenerator> {
  llvm::raw_ostream &Out;
public:
  USRGenerator(llvm::raw_ostream &out) : Out(out) {}

  void VisitObjCContainerDecl(ObjCContainerDecl *CD);  
  void VisitObjCMethodDecl(ObjCMethodDecl *MD);
  void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
};
} // end anonymous namespace

void USRGenerator::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  Visit(cast<Decl>(D->getDeclContext()));
  Out << (D->isInstanceMethod() ? "_IM_" : "_CM_");
  Out << DeclarationName(D->getSelector());
}
  
void USRGenerator::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  switch (D->getKind()) {
    default:
      assert(false && "Invalid ObjC container.");
    case Decl::ObjCInterface:
    case Decl::ObjCImplementation:
      Out << "objc_class_" << D->getName();
      break;
    case Decl::ObjCCategory: {
      ObjCCategoryDecl *CD = cast<ObjCCategoryDecl>(D);
      Out << "objc_cat_" << CD->getClassInterface()->getName()
          << '_' << CD->getName();
      break;
    }
    case Decl::ObjCCategoryImpl: {
      ObjCCategoryImplDecl *CD = cast<ObjCCategoryImplDecl>(D);
      Out << "objc_cat_" << CD->getClassInterface()->getName()
          << '_' << CD->getName();
      break;
    }
    case Decl::ObjCProtocol:
      Out << "objc_prot_" << cast<ObjCProtocolDecl>(D)->getName();
      break;
  }
}
  
void USRGenerator::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  Visit(cast<Decl>(D->getDeclContext()));
  Out << "_prop_" << D->getName();
}
  
// FIXME: This is a skeleton implementation.  It will be overhauled.
CXString clang_getUSR(CXEntity CE) {
  const Entity &E = GetEntity(CE);
  
  // FIXME: Support cross-translation unit CXEntities.  
  if (!E.isInternalToTU())
    return CIndexer::createCXString(NULL);
  
  Decl *D = E.getInternalDecl();
  if (!D)
    return CIndexer::createCXString(NULL);

  llvm::SmallString<1024> StrBuf;
  {
    llvm::raw_svector_ostream Out(StrBuf);
    USRGenerator UG(Out);
    UG.Visit(D);
  }
  
  if (StrBuf.empty())
    return CIndexer::createCXString(NULL);

  // Return a copy of the string that must be disposed by the caller.
  return CIndexer::createCXString(StrBuf.c_str(), true);
}

} // end extern "C"
