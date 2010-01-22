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
#include "CXCursor.h"
#include "clang/AST/DeclVisitor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// USR generation.
//===----------------------------------------------------------------------===//

namespace {
class USRGenerator : public DeclVisitor<USRGenerator> {
  llvm::raw_ostream &Out;
  bool IgnoreResults;
public:
  USRGenerator(llvm::raw_ostream &out) : Out(out), IgnoreResults(false) {}
  
  bool ignoreResults() const { return IgnoreResults; }
  
  void VisitBlockDecl(BlockDecl *D);
  void VisitDeclContext(DeclContext *D);
  void VisitFieldDecl(FieldDecl *D);
  void VisitFunctionDecl(FunctionDecl *D);
  void VisitNamedDecl(NamedDecl *D);
  void VisitNamespaceDecl(NamespaceDecl *D);
  void VisitObjCContainerDecl(ObjCContainerDecl *CD);  
  void VisitObjCMethodDecl(ObjCMethodDecl *MD);
  void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
  void VisitTagDecl(TagDecl *D);
  void VisitTypedefDecl(TypedefDecl *D);
};
} // end anonymous namespace

void USRGenerator::VisitBlockDecl(BlockDecl *D) {
  VisitDeclContext(D->getDeclContext());
  // FIXME: Better support for anonymous blocks.
  Out << "@B^anon";
}

void USRGenerator::VisitDeclContext(DeclContext *DC) {
  if (NamedDecl *D = dyn_cast<NamedDecl>(DC))
    Visit(D);
}

void USRGenerator::VisitFieldDecl(FieldDecl *D) {
  const std::string &s = D->getNameAsString();
  if (s.empty()) {
    // Bit fields can be anonymous.
    IgnoreResults = true;
    return;
  }
  VisitDeclContext(D->getDeclContext());
  Out << "@^FI^" << s;
}

void USRGenerator::VisitFunctionDecl(FunctionDecl *D) {
  VisitDeclContext(D->getDeclContext());
  Out << "@F^" << D->getNameAsString();
}

void USRGenerator::VisitNamedDecl(NamedDecl *D) {
  VisitDeclContext(D->getDeclContext());
  const std::string &s = D->getNameAsString();
  assert(!s.empty());
  Out << "@^" << s;
}

void USRGenerator::VisitNamespaceDecl(NamespaceDecl *D) {
  VisitDeclContext(D->getDeclContext());
  Out << "@N^" << D->getNameAsString();
}

void USRGenerator::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  Visit(cast<Decl>(D->getDeclContext()));
  Out << (D->isInstanceMethod() ? "(im)" : "(cm)");
  Out << DeclarationName(D->getSelector()).getAsString();
}

void USRGenerator::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  switch (D->getKind()) {
    default:
      assert(false && "Invalid ObjC container.");
    case Decl::ObjCInterface:
    case Decl::ObjCImplementation:
      Out << "objc(cs)" << D->getName();
      break;
    case Decl::ObjCCategory: {
      ObjCCategoryDecl *CD = cast<ObjCCategoryDecl>(D);
      Out << "objc(cy)" << CD->getClassInterface()->getName()
      << '^' << CD->getName();
      break;
    }
    case Decl::ObjCCategoryImpl: {
      ObjCCategoryImplDecl *CD = cast<ObjCCategoryImplDecl>(D);
      Out << "objc(cy)" << CD->getClassInterface()->getName()
      << '^' << CD->getName();
      break;
    }
    case Decl::ObjCProtocol:
      Out << "objc(pl)" << cast<ObjCProtocolDecl>(D)->getName();
      break;
  }
}

void USRGenerator::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  Visit(cast<Decl>(D->getDeclContext()));
  Out << "(py)" << D->getName();
}

void USRGenerator::VisitTagDecl(TagDecl *D) {
  VisitDeclContext(D->getDeclContext());
  switch (D->getTagKind()) {
    case TagDecl::TK_struct: Out << "@S^"; break;
    case TagDecl::TK_class:  Out << "@C^"; break;
    case TagDecl::TK_union:  Out << "@U^"; break;
    case TagDecl::TK_enum:   Out << "@E^"; break;
  }
  
  // FIXME: Better support for anonymous structures and enums.
  const std::string &s = D->getNameAsString();
  if (s.empty()) {
    if (TypedefDecl *TD = D->getTypedefForAnonDecl())
      Out << "^anontd^" << TD->getNameAsString();    
    else
      Out << "^anon";
  }
  else
    Out << s;
}

void USRGenerator::VisitTypedefDecl(TypedefDecl *D) {
  DeclContext *DC = D->getDeclContext();
  if (NamedDecl *DCN = dyn_cast<NamedDecl>(DC))
    Visit(DCN);  
  Out << "typedef@" << D->getName();
}

// FIXME: This is a skeleton implementation.  It will be overhauled.
static CXString ConstructUSR(Decl *D) {
  llvm::SmallString<1024> StrBuf;
  {
    llvm::raw_svector_ostream Out(StrBuf);
    USRGenerator UG(Out);
    UG.Visit(static_cast<Decl*>(D));
    if (UG.ignoreResults())
      return CIndexer::createCXString(NULL);
  }
  
  if (StrBuf.empty())
    return CIndexer::createCXString(NULL);
  
  // Return a copy of the string that must be disposed by the caller.
  return CIndexer::createCXString(StrBuf.c_str(), true);
}  


extern "C" {

CXString clang_getCursorUSR(CXCursor C) {
  if (Decl *D = cxcursor::getCursorDecl(C))
    return ConstructUSR(D);  
  
  return CIndexer::createCXString(NULL);
}

} // end extern "C"
