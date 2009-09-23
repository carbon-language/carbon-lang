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

    if (Cursor.kind == CXCursor_FunctionDefn) {
      const char *startBuf, *endBuf;
      unsigned startLine, startColumn, endLine, endColumn;
      clang_getDefinitionSpellingAndExtent(Cursor, &startBuf, &endBuf,
                                           &startLine, &startColumn,
                                           &endLine, &endColumn);
      {
        /* Probe the entire body, looking for both decls and refs. */
        unsigned curLine = startLine, curColumn = startColumn;
        CXCursor Ref;
        
        while (startBuf <= endBuf) {
          if (*startBuf == '\n') {
            startBuf++;
            curLine++;
            curColumn = 1;
          } else if (*startBuf != '\t')
            curColumn++;
        
          Ref = clang_getCursor(Unit, clang_getCursorSource(Cursor), 
                                curLine, curColumn);
          if (Ref.kind != CXCursor_FunctionDecl) {
            PrintCursor(Ref);
            printf("(Context: %s", clang_getDeclSpelling(Ref.decl));
            printf(" Source:  %s (%d:%d))\n", clang_getCursorSource(Ref),
                                              curLine, curColumn);
          }
          startBuf++;
        }
      }
    }
    clang_loadDeclaration(Cursor.decl, DeclVisitor, 0);
  }
}

/*
 * First sign of life:-)
 */
int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Incorrect usage of c-index-test (requires 3 arguments)\n");
    return 0;
  }
  {
  CXIndex Idx = clang_createIndex();
  CXTranslationUnit TU = clang_createTranslationUnit(Idx, argv[1]);
  enum CXCursorKind K = CXCursor_NotImplemented;
  
  if (!strcmp(argv[2], "all")) {
    clang_loadTranslationUnit(TU, TranslationUnitVisitor, 0);
    return 1;
  } 
  /* Perform some simple filtering. */
  if (!strcmp(argv[2], "category")) K = CXCursor_ObjCCategoryDecl;
  else if (!strcmp(argv[2], "interface")) K = CXCursor_ObjCInterfaceDecl;
  else if (!strcmp(argv[2], "protocol")) K = CXCursor_ObjCProtocolDecl;
  else if (!strcmp(argv[2], "function")) K = CXCursor_FunctionDecl;
  else if (!strcmp(argv[2], "typedef")) K = CXCursor_TypedefDecl;
  
  clang_loadTranslationUnit(TU, TranslationUnitVisitor, &K);
  return 1;
  }
}
