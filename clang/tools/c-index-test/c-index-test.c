/* c-index-test.c */

#include "clang-c/Index.h"
#include <stdio.h>

static void DeclVisitor(CXDecl Dcl, CXCursor Cursor, CXClientData Filter) 
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    printf("%s => %s", clang_getCursorKindSpelling(Cursor.kind),
                       clang_getCursorSpelling(Cursor));
    printf(" (%s,%d:%d)\n", clang_getCursorSource(Cursor),
                            clang_getCursorLine(Cursor),
                            clang_getCursorColumn(Cursor));
  }
}
static void TranslationUnitVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                                   CXClientData Filter) 
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    printf("%s => %s", clang_getCursorKindSpelling(Cursor.kind),
                       clang_getCursorSpelling(Cursor));
    printf(" (%s,%d:%d)\n", clang_getCursorSource(Cursor),
                            clang_getCursorLine(Cursor),
                            clang_getCursorColumn(Cursor));

    enum CXCursorKind filterData = CXCursor_FieldDecl;
    clang_loadDeclaration(Cursor.decl, DeclVisitor, 0);
  }
}

/*
 * First sign of life:-)
 */
int main(int argc, char **argv) {
  CXIndex Idx = clang_createIndex();
  CXTranslationUnit TU = clang_createTranslationUnit(Idx, argv[1]);
  
  enum CXCursorKind filterData = CXCursor_StructDecl;
  clang_loadTranslationUnit(TU, TranslationUnitVisitor, 0);
  return 1;
}
