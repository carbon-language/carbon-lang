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

#include "clang/Frontend/ASTUnit.h"
#include "clang/Basic/FileManager.h"

#include "clang/AST/DeclVisitor.h"

using namespace clang;
using namespace idx;

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

namespace {

class IdxVisitor : public DeclVisitor<IdxVisitor> {
public:
  void VisitNamedDecl(NamedDecl *ND) {
    printf("NamedDecl (%s:", ND->getDeclKindName());
    if (ND->getIdentifier())
      printf("%s)\n", ND->getIdentifier()->getName());
    else
      printf("<no name>)\n");
  }
};

}

void clang_loadTranslationUnit(
  CXTranslationUnit CTUnit, void (*callback)(CXTranslationUnit, CXCursor))
{
  assert(CTUnit && "Passed null CXTranslationUnit");
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(CTUnit);
  ASTContext &Ctx = CXXUnit->getASTContext();
  
  IdxVisitor DVisit;
  DVisit.Visit(Ctx.getTranslationUnitDecl());
}

void clang_loadDeclaration(CXDecl, void (*callback)(CXDecl, CXCursor))
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
  return 0;
}
CXEntity clang_getEntityFromDecl(CXDecl)
{
  return 0;
}
enum CXDeclKind clang_getDeclKind(CXDecl)
{
  return CXDecl_any;
}
const char *clang_getDeclSpelling(CXDecl)
{
  return "";
}
//
// CXCursor Operations.
//
CXCursor clang_getCursor(CXTranslationUnit, const char *source_name, 
                         unsigned line, unsigned column)
{
  return 0;
}

CXCursorKind clang_getCursorKind(CXCursor)
{
  return CXCursor_Declaration;
}

unsigned clang_getCursorLine(CXCursor)
{
  return 0;
}
unsigned clang_getCursorColumn(CXCursor)
{
  return 0;
}
const char *clang_getCursorSource(CXCursor) 
{
  return "";
}

// If CXCursorKind == Cursor_Reference, then this will return the referenced declaration.
// If CXCursorKind == Cursor_Declaration, then this will return the declaration.
CXDecl clang_getCursorDecl(CXCursor) 
{
  return 0;
}

} // end extern "C"
