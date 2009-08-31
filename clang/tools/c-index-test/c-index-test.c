
#include "clang-c/Index.h"
#include <stdio.h>

static void PrintDecls(CXTranslationUnit Unit, CXCursor Cursor) {
  if (clang_isDeclaration(Cursor.kind)) {
    printf("%s => %s", clang_getKindSpelling(Cursor.kind),
                       clang_getDeclSpelling(Cursor.decl));
    printf(" (%s,%d:%d)\n", clang_getCursorSource(Cursor),
                            clang_getCursorLine(Cursor),
                            clang_getCursorColumn(Cursor));
  }
}

/*
 * First sign of life:-)
 */
int main(int argc, char **argv) {
  CXIndex Idx = clang_createIndex();
  CXTranslationUnit TU = clang_createTranslationUnit(Idx, argv[1]);
  clang_loadTranslationUnit(TU, PrintDecls);
  return 1;
}
