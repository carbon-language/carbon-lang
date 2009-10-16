/* c-index-test.c */

#include "clang-c/Index.h"
#include <stdio.h>
#include <string.h>

extern char *basename(const char *);

static void PrintCursor(CXCursor Cursor) {
  if (clang_isInvalid(Cursor.kind))
    printf("Invalid Cursor => %s\n", clang_getCursorKindSpelling(Cursor.kind));
  else {
    CXDecl DeclReferenced;
    printf("%s=%s", clang_getCursorKindSpelling(Cursor.kind),
                      clang_getCursorSpelling(Cursor));
    DeclReferenced = clang_getCursorDecl(Cursor);
    if (DeclReferenced)
      printf(":%d:%d", clang_getDeclLine(DeclReferenced),
                       clang_getDeclColumn(DeclReferenced));
  }
}

static void DeclVisitor(CXDecl Dcl, CXCursor Cursor, CXClientData Filter)
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    printf("// CHECK: %s:%d:%d: ", basename(clang_getCursorSource(Cursor)),
                                 clang_getCursorLine(Cursor),
                                 clang_getCursorColumn(Cursor));
    PrintCursor(Cursor);
    printf(" [Context=%s]\n", clang_getDeclSpelling(Dcl));
  }
}
static void TranslationUnitVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                                   CXClientData Filter)
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    printf("// CHECK: %s:%d:%d: ", basename(clang_getCursorSource(Cursor)),
                                 clang_getCursorLine(Cursor),
                                 clang_getCursorColumn(Cursor));
    PrintCursor(Cursor);
    printf(" [Context=%s]\n", basename(clang_getTranslationUnitSpelling(Unit)));

    clang_loadDeclaration(Cursor.decl, DeclVisitor, 0);

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
            printf("// CHECK: %s:%d:%d: ", basename(clang_getCursorSource(Ref)),
                                             curLine, curColumn);
            PrintCursor(Ref);
            printf(" [Context:%s]\n", clang_getDeclSpelling(Ref.decl));
          }
          startBuf++;
        }
      }
    }
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
  if (!strcmp(argv[2], "local"))
    clang_wantOnlyLocalDeclarations(Idx);
  CXTranslationUnit TU = clang_createTranslationUnit(Idx, argv[1]);
  if (!TU) {
    fprintf(stderr, "Unable to load translation unit!\n");
    return 1;
  }
  enum CXCursorKind K = CXCursor_NotImplemented;

  if (!strcmp(argv[2], "all") || !strcmp(argv[2], "local")) {
    clang_loadTranslationUnit(TU, TranslationUnitVisitor, 0);
    clang_disposeTranslationUnit(TU);
    return 1;
  }
  /* Perform some simple filtering. */
  if (!strcmp(argv[2], "category")) K = CXCursor_ObjCCategoryDecl;
  else if (!strcmp(argv[2], "interface")) K = CXCursor_ObjCInterfaceDecl;
  else if (!strcmp(argv[2], "protocol")) K = CXCursor_ObjCProtocolDecl;
  else if (!strcmp(argv[2], "function")) K = CXCursor_FunctionDecl;
  else if (!strcmp(argv[2], "typedef")) K = CXCursor_TypedefDecl;

  clang_loadTranslationUnit(TU, TranslationUnitVisitor, &K);
  clang_disposeTranslationUnit(TU);
  return 1;
  }
}
