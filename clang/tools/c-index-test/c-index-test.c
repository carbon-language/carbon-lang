/* c-index-test.c */

#include "clang-c/Index.h"
#include <stdio.h>
#include <string.h>

static void PrintCursor(CXCursor Cursor) {
  if (clang_isInvalid(Cursor.kind))
    printf("Invalid Cursor => %s\n", clang_getCursorKindSpelling(Cursor.kind));
  else
    printf("%s => %s ", clang_getCursorKindSpelling(Cursor.kind),
                        clang_getCursorSpelling(Cursor));
}

static void DeclVisitor(CXDecl Dcl, CXCursor Cursor, CXClientData Filter) 
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    PrintCursor(Cursor);
    printf("(Context: %s", clang_getDeclSpelling(Dcl));
    printf(" Source:  %s (%d:%d))\n", clang_getCursorSource(Cursor),
                                      clang_getCursorLine(Cursor),
                                      clang_getCursorColumn(Cursor));
  }
}
static void TranslationUnitVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                                   CXClientData Filter) 
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    PrintCursor(Cursor);
    printf("(Context: %s", clang_getTranslationUnitSpelling(Unit));
    printf(" Source:  %s (%d:%d))\n", clang_getCursorSource(Cursor),
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

  if (argc == 2) {
    /* malloc - returns a cursor of type CXCursor_FunctionDecl */
    CXCursor C = clang_getCursor(TU, "/usr/include/stdlib.h", 169, 7);
    PrintCursor(C);
    /* methodSignature - returns a cursor of type ObjCInstanceMethodDecl */
    C = clang_getCursor(TU, "/System/Library/Frameworks/Foundation.framework/Headers/NSInvocation.h", 22, 1);
    PrintCursor(C);
    C = clang_getCursor(TU, "Large.m", 5, 18);
    PrintCursor(C);
  } else if (argc == 3) {
    enum CXCursorKind K = CXCursor_NotImplemented;
    
    if (!strcmp(argv[2], "all")) {
      clang_loadTranslationUnit(TU, TranslationUnitVisitor, 0);
      return 1;
    } 
    if (!strcmp(argv[2], "category")) K = CXCursor_ObjCCategoryDecl;
    else if (!strcmp(argv[2], "interface")) K = CXCursor_ObjCInterfaceDecl;
    else if (!strcmp(argv[2], "protocol")) K = CXCursor_ObjCProtocolDecl;
    else if (!strcmp(argv[2], "function")) K = CXCursor_FunctionDecl;
    else if (!strcmp(argv[2], "typedef")) K = CXCursor_TypedefDecl;
    
    clang_loadTranslationUnit(TU, TranslationUnitVisitor, &K);
  }
  return 1;
}
