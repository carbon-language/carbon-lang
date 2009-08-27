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

extern "C" {

// Some notes on CXEntity:
//
// - Since the 'ordinary' namespace includes functions, data, typedefs, ObjC interfaces, the
// current algorithm is a bit naive (resulting in one entity for 2 different types). For example:
//
// module1.m: @interface Foo @end Foo *x;
// module2.m: void Foo(int);
//
// - Since the unique name spans translation units, static data/functions within a CXTranslationUnit
// are *not* currently represented by entities. As a result, there will be no entity for the following:
//
// module.m: static void Foo() { }
//

CXIndex clang_createIndex() 
{ 
  return 0; 
}

CXTranslationUnit clang_loadTranslationUnitFromASTFile(
  CXIndex, const char *ast_filename) 
{
  return 0;
}

void clang_loadTranslationUnit(
  CXTranslationUnit, void (*callback)(CXTranslationUnit, CXCursor)
)
{
}

void clang_loadDeclaration(CXDecl, void (*callback)(CXDecl, CXCursor))
{
}

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
  return Cursor_Declaration;
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
