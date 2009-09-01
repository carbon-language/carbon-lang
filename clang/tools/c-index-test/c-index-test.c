/* c-index-test.c */

#include "clang-c/Index.h"
#include <stdio.h>

static void PrintDecls(CXTranslationUnit Unit, CXCursor Cursor,
                       CXClientData Filter) {
  if (clang_isDeclaration(Cursor.kind)) {
    if (Cursor.kind == *(enum CXCursorKind *)Filter) {
      printf("%s => %s", clang_getKindSpelling(Cursor.kind),
                         clang_getDeclSpelling(Cursor.decl));
      printf(" (%s,%d:%d)\n", clang_getCursorSource(Cursor),
                              clang_getCursorLine(Cursor),
                              clang_getCursorColumn(Cursor));
    }
  }
}

/*
 * First sign of life:-)
 */
int main(int argc, char **argv) {
  CXIndex Idx = clang_createIndex();
  CXTranslationUnit TU = clang_createTranslationUnit(Idx, argv[1]);
  
  /* Use client data to only print ObjC interfaces */
  enum CXCursorKind filterData = CXCursor_ObjCInterfaceDecl;
  clang_loadTranslationUnit(TU, PrintDecls, &filterData);
  return 1;
}
