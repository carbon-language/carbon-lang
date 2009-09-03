/* c-index-test.c */

#include "clang-c/Index.h"
#include <stdio.h>
#include <string.h>

static void PrintCursor(CXCursor Cursor) {
  printf("%s => %s\n", clang_getCursorKindSpelling(Cursor.kind),
                       clang_getCursorSpelling(Cursor));
}

static void DeclVisitor(CXDecl Dcl, CXCursor Cursor, CXClientData Filter) 
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    PrintCursor(Cursor);
    printf("  Context: %s\n", clang_getDeclSpelling(Dcl));
    printf("  Source:  %s (%d:%d)\n", clang_getCursorSource(Cursor),
                                      clang_getCursorLine(Cursor),
                                      clang_getCursorColumn(Cursor));
  }
}
static void TranslationUnitVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                                   CXClientData Filter) 
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    PrintCursor(Cursor);
    printf("  Context: %s\n", clang_getTranslationUnitSpelling(Unit));
    printf("  Source:  %s (%d:%d)\n", clang_getCursorSource(Cursor),
                                      clang_getCursorLine(Cursor),
                                      clang_getCursorColumn(Cursor));

    clang_loadDeclaration(Cursor.decl, DeclVisitor, 0);
  }
}

/*
 * First sign of life:-)
 */
int main(int argc, char **argv) {
  CXIndex Idx = clang_createIndex();
  CXTranslationUnit TU = clang_createTranslationUnit(Idx, argv[1]);
  
  if (argc == 2)
    clang_loadTranslationUnit(TU, TranslationUnitVisitor, 0);
  else if (argc == 3) {
    enum CXCursorKind K = CXCursor_Invalid;
    
    if (!strcmp(argv[2], "category")) K = CXCursor_ObjCCategoryDecl;
    else if (!strcmp(argv[2], "interface")) K = CXCursor_ObjCInterfaceDecl;
    else if (!strcmp(argv[2], "protocol")) K = CXCursor_ObjCProtocolDecl;
    else if (!strcmp(argv[2], "function")) K = CXCursor_FunctionDecl;
    else if (!strcmp(argv[2], "typedef")) K = CXCursor_TypedefDecl;
    
    clang_loadTranslationUnit(TU, TranslationUnitVisitor, &K);
  }
  return 1;
}
