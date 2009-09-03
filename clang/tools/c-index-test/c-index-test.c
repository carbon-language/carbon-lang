/* c-index-test.c */

#include "clang-c/Index.h"
#include <stdio.h>

static void PrintCursor(CXCursor Cursor) {
  printf("%s => %s", clang_getCursorKindSpelling(Cursor.kind),
                     clang_getCursorSpelling(Cursor));
  printf(" (%s,%d:%d)\n", clang_getCursorSource(Cursor),
                          clang_getCursorLine(Cursor),
                          clang_getCursorColumn(Cursor));
}

static void DeclVisitor(CXDecl Dcl, CXCursor Cursor, CXClientData Filter) 
{
  printf("%s: ", clang_getDeclSpelling(Dcl));
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter))
    PrintCursor(Cursor);
}

static void TranslationUnitVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                                   CXClientData Filter) 
{
  printf("%s: ", clang_getTranslationUnitSpelling(Unit));
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    PrintCursor(Cursor);

    clang_loadDeclaration(Cursor.decl, DeclVisitor, 0);
  }
}

/*
 * First sign of life:-)
 */
int main(int argc, char **argv) {
  CXIndex Idx = clang_createIndex();
  CXTranslationUnit TU = clang_createTranslationUnit(Idx, argv[1]);
  
  clang_loadTranslationUnit(TU, TranslationUnitVisitor, 0);
  return 1;
}
