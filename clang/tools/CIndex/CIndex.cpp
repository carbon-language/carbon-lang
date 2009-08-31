//===- CIndex.cpp - Clang-C Source Indexing Library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Clang-C Source Indexing library.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Index.h"
#include "clang/Index/Program.h"
#include "clang/Index/Indexer.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/ASTUnit.h"
#include <cstdio>
using namespace clang;
using namespace idx;

namespace {

// Translation Unit Visitor.
class TUVisitor : public DeclVisitor<TUVisitor> {
  CXTranslationUnit TUnit;
  CXTranslationUnitIterator Callback;
public:
  TUVisitor(CXTranslationUnit CTU, CXTranslationUnitIterator cback) : 
    TUnit(CTU), Callback(cback) {}
  
  void VisitTranslationUnitDecl(TranslationUnitDecl *D) {
    VisitDeclContext(dyn_cast<DeclContext>(D));
  }
  void VisitDeclContext(DeclContext *DC) {
    for (DeclContext::decl_iterator
           I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitTypedefDecl(TypedefDecl *ND) {
    CXCursor C = { CXCursor_TypedefDecl, ND };
    Callback(TUnit, C);
  }
  void VisitTagDecl(TagDecl *ND) {
    CXCursor C = { ND->isEnum() ? CXCursor_EnumDecl : CXCursor_RecordDecl, ND };
    Callback(TUnit, C);
  }
  void VisitFunctionDecl(FunctionDecl *ND) {
    CXCursor C = { CXCursor_FunctionDecl, ND };
    Callback(TUnit, C);
  }
  void VisitObjCInterfaceDecl(ObjCInterfaceDecl *ND) {
    CXCursor C = { CXCursor_ObjCInterfaceDecl, ND };
    Callback(TUnit, C);
  }
  void VisitObjCCategoryDecl(ObjCCategoryDecl *ND) {
    CXCursor C = { CXCursor_ObjCCategoryDecl, ND };
    Callback(TUnit, C);
  }
  void VisitObjCProtocolDecl(ObjCProtocolDecl *ND) {
    CXCursor C = { CXCursor_ObjCProtocolDecl, ND };
    Callback(TUnit, C);
  }
};

// Top-level declaration visitor.
class TLDeclVisitor : public DeclVisitor<TLDeclVisitor> {
public:
  void VisitDeclContext(DeclContext *DC) {
    for (DeclContext::decl_iterator
           I = DC->decls_begin(), E = DC->decls_end(); I != E; ++I)
      Visit(*I);
  }
  void VisitEnumConstantDecl(EnumConstantDecl *ND) {
  }
  void VisitFieldDecl(FieldDecl *ND) {
  }
  void VisitObjCIvarDecl(ObjCIvarDecl *ND) {
  }
};

}

extern "C" {

CXIndex clang_createIndex() 
{
  return new Indexer(*new Program(), *new FileManager());
}

// FIXME: need to pass back error info.
CXTranslationUnit clang_createTranslationUnit(
  CXIndex CIdx, const char *ast_filename) 
{
  assert(CIdx && "Passed null CXIndex");
  Indexer *CXXIdx = static_cast<Indexer *>(CIdx);
  std::string astName(ast_filename);
  std::string ErrMsg;
  
  return ASTUnit::LoadFromPCHFile(astName, CXXIdx->getFileManager(), &ErrMsg);
}


void clang_loadTranslationUnit(
  CXTranslationUnit CTUnit, CXTranslationUnitIterator callback)
{
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);
  ASTContext &Ctx = CXXUnit->getASTContext();
  
  TUVisitor DVisit(CTUnit, callback);
  DVisit.Visit(Ctx.getTranslationUnitDecl());
}

void clang_loadDeclaration(CXDecl, CXDeclIterator)
{
}

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


const char *clang_getDeclarationName(CXEntity)
{
  return "";
}
const char *clang_getURI(CXEntity)
{
  return "";
}

CXEntity clang_getEntity(const char *URI)
{
  return 0;
}

//
// CXDecl Operations.
//
CXCursor clang_getCursorFromDecl(CXDecl)
{
  return CXCursor();
}
CXEntity clang_getEntityFromDecl(CXDecl)
{
  return 0;
}
const char *clang_getDeclSpelling(CXDecl AnonDecl)
{
  assert(AnonDecl && "Passed null CXDecl");
  NamedDecl *ND = static_cast<NamedDecl *>(AnonDecl);
  if (ND->getIdentifier())
    return ND->getIdentifier()->getName();
  else
    return "";
}
const char *clang_getKindSpelling(enum CXCursorKind Kind)
{
  switch (Kind) {
   case CXCursor_FunctionDecl: return "FunctionDecl";
   case CXCursor_TypedefDecl: return "TypedefDecl";
   case CXCursor_EnumDecl: return "EnumDecl";
   case CXCursor_EnumConstantDecl: return "EnumConstantDecl";
   case CXCursor_RecordDecl: return "RecordDecl";
   case CXCursor_FieldDecl: return "FieldDecl";
   case CXCursor_VarDecl: return "VarDecl";
   case CXCursor_ParmDecl: return "ParmDecl";
   case CXCursor_ObjCInterfaceDecl: return "ObjCInterfaceDecl";
   case CXCursor_ObjCCategoryDecl: return "ObjCCategoryDecl";
   case CXCursor_ObjCProtocolDecl: return "ObjCProtocolDecl";
   case CXCursor_ObjCPropertyDecl: return "ObjCPropertyDecl";
   case CXCursor_ObjCIvarDecl: return "ObjCIvarDecl";
   case CXCursor_ObjCMethodDecl: return "ObjCMethodDecl";
   default: return "<not implemented>";
  }
}

//
// CXCursor Operations.
//
CXCursor clang_getCursor(CXTranslationUnit, const char *source_name, 
                         unsigned line, unsigned column)
{
  return CXCursor();
}

CXCursorKind clang_getCursorKind(CXCursor)
{
  return CXCursor_Invalid;
}

unsigned clang_isDeclaration(enum CXCursorKind K)
{
  return K >= CXCursor_FirstDecl && K <= CXCursor_LastDecl;
}

unsigned clang_getCursorLine(CXCursor C)
{
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceLocation SLoc = ND->getLocation();
  if (SLoc.isInvalid())
    return 0;
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  SLoc = SourceMgr.getSpellingLoc(SLoc); // handles macro instantiations.
  return SourceMgr.getSpellingLineNumber(SLoc);
}
unsigned clang_getCursorColumn(CXCursor C)
{
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceLocation SLoc = ND->getLocation();
  if (SLoc.isInvalid())
    return 0;
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  SLoc = SourceMgr.getSpellingLoc(SLoc); // handles macro instantiations.
  return SourceMgr.getSpellingColumnNumber(SLoc);
}
const char *clang_getCursorSource(CXCursor C) 
{
  assert(C.decl && "CXCursor has null decl");
  NamedDecl *ND = static_cast<NamedDecl *>(C.decl);
  SourceLocation SLoc = ND->getLocation();
  if (SLoc.isInvalid())
    return "<invalid source location>";
  SourceManager &SourceMgr = ND->getASTContext().getSourceManager();
  SLoc = SourceMgr.getSpellingLoc(SLoc); // handles macro instantiations.
  return SourceMgr.getBufferName(SLoc);
}

// If CXCursorKind == Cursor_Reference, then this will return the referenced declaration.
// If CXCursorKind == Cursor_Declaration, then this will return the declaration.
CXDecl clang_getCursorDecl(CXCursor) 
{
  return 0;
}

} // end extern "C"
