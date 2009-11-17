/* c-index-test.c */

#include "clang-c/Index.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/******************************************************************************/
/* Utility functions.                                                         */
/******************************************************************************/

#ifdef _MSC_VER
char *basename(const char* path)
{
    char* base1 = (char*)strrchr(path, '/');
    char* base2 = (char*)strrchr(path, '\\');
    if (base1 && base2)
        return((base1 > base2) ? base1 + 1 : base2 + 1);
    else if (base1)
        return(base1 + 1);
    else if (base2)
        return(base2 + 1);

    return((char*)path);
}
#else
extern char *basename(const char *);
#endif

/******************************************************************************/
/* Pretty-printing.                                                           */
/******************************************************************************/

static void PrintCursor(CXCursor Cursor) {
  if (clang_isInvalid(Cursor.kind))
    printf("Invalid Cursor => %s\n", clang_getCursorKindSpelling(Cursor.kind));
  else {
    CXDecl DeclReferenced;
    CXString string;
    string = clang_getCursorSpelling(Cursor);
    printf("%s=%s", clang_getCursorKindSpelling(Cursor.kind),
                      clang_getCString(string));
    clang_disposeString(string);
    DeclReferenced = clang_getCursorDecl(Cursor);
    if (DeclReferenced)
      printf(":%d:%d", clang_getDeclLine(DeclReferenced),
                       clang_getDeclColumn(DeclReferenced));
  }
}

static const char* GetCursorSource(CXCursor Cursor) {  
  const char *source = clang_getCursorSource(Cursor);
  if (!source)
    return "<invalid loc>";  
  return basename(source);
}

/******************************************************************************/
/* Logic for testing clang_loadTranslationUnit().                             */
/******************************************************************************/

static void DeclVisitor(CXDecl Dcl, CXCursor Cursor, CXClientData Filter)
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    CXString string;
    printf("// CHECK: %s:%d:%d: ", GetCursorSource(Cursor),
                                 clang_getCursorLine(Cursor),
                                 clang_getCursorColumn(Cursor));
    PrintCursor(Cursor);
    string = clang_getDeclSpelling(Dcl);
    printf(" [Context=%s]\n", clang_getCString(string));
    clang_disposeString(string);
  }
}
static void TranslationUnitVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                                   CXClientData Filter)
{
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    CXString string;
    printf("// CHECK: %s:%d:%d: ", GetCursorSource(Cursor),
                                 clang_getCursorLine(Cursor),
                                 clang_getCursorColumn(Cursor));
    PrintCursor(Cursor);
    string = clang_getTranslationUnitSpelling(Unit);
    printf(" [Context=%s]\n",
          basename(clang_getCString(string)));
    clang_disposeString(string);

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

        while (startBuf < endBuf) {
          if (*startBuf == '\n') {
            startBuf++;
            curLine++;
            curColumn = 1;
          } else if (*startBuf != '\t')
            curColumn++;
          
          Ref = clang_getCursor(Unit, clang_getCursorSource(Cursor),
                                curLine, curColumn);
          if (Ref.kind == CXCursor_NoDeclFound) {
            /* Nothing found here; that's fine. */
          } else if (Ref.kind != CXCursor_FunctionDecl) {
            CXString string;
            printf("// CHECK: %s:%d:%d: ", GetCursorSource(Ref),
                   curLine, curColumn);
            PrintCursor(Ref);
            string = clang_getDeclSpelling(Ref.decl);
            printf(" [Context:%s]\n", clang_getCString(string));
            clang_disposeString(string);
          }
          startBuf++;
        }
      }
    }
  }
}

int perform_test_load_tu(const char *file, const char *filter) {
  CXIndex Idx;
  CXTranslationUnit TU;
  enum CXCursorKind K = CXCursor_NotImplemented;
  enum CXCursorKind *ck = &K;
  Idx = clang_createIndex(/* excludeDeclsFromPCH */ 
                          !strcmp(filter, "local") ? 1 : 0, 
                          /* displayDiagnostics */ 1);
  
  TU = clang_createTranslationUnit(Idx, file);
  
  if (!TU) {
    fprintf(stderr, "Unable to load translation unit from '%s'!\n", file);
    return 1;
  }
  
  /* Perform some simple filtering. */
  if (!strcmp(filter, "all") || !strcmp(filter, "local")) ck = NULL;
  else if (!strcmp(filter, "category")) K = CXCursor_ObjCCategoryDecl;
  else if (!strcmp(filter, "interface")) K = CXCursor_ObjCInterfaceDecl;
  else if (!strcmp(filter, "protocol")) K = CXCursor_ObjCProtocolDecl;
  else if (!strcmp(filter, "function")) K = CXCursor_FunctionDecl;
  else if (!strcmp(filter, "typedef")) K = CXCursor_TypedefDecl;
  else {
    fprintf(stderr, "Unknown filter for -test-load-tu: %s\n", filter);
    return 1;
  }
  
  clang_loadTranslationUnit(TU, TranslationUnitVisitor, ck);
  clang_disposeTranslationUnit(TU);
  return 0;
}

/******************************************************************************/
/* Logic for testing clang_codeComplete().                                    */
/******************************************************************************/

/* Parse file:line:column from the input string. Returns 0 on success, non-zero
   on failure. If successful, the pointer *filename will contain newly-allocated
   memory (that will be owned by the caller) to store the file name. */
int parse_file_line_column(const char *input, char **filename, unsigned *line, 
                           unsigned *column) {
  /* Find the second colon. */
  const char *second_colon = strrchr(input, ':'), *first_colon;
  char *endptr = 0;
  if (!second_colon || second_colon == input) {
    fprintf(stderr, "could not parse filename:line:column in '%s'\n", input);
    return 1;
  }

  /* Parse the column number. */
  *column = strtol(second_colon + 1, &endptr, 10);
  if (*endptr != 0) {
    fprintf(stderr, "could not parse column in '%s'\n", input);
    return 1;
  }

  /* Find the first colon. */
  first_colon = second_colon - 1;
  while (first_colon != input && *first_colon != ':')
    --first_colon;
  if (first_colon == input) {
    fprintf(stderr, "could not parse line in '%s'\n", input);
    return 1;    
  }

  /* Parse the line number. */
  *line = strtol(first_colon + 1, &endptr, 10);
  if (*endptr != ':') {
    fprintf(stderr, "could not parse line in '%s'\n", input);
    return 1;
  }
  
  /* Copy the file name. */
  *filename = (char*)malloc(first_colon - input + 1);
  memcpy(*filename, input, first_colon - input);
  (*filename)[first_colon - input] = 0;
  return 0;
}

const char *
clang_getCompletionChunkKindSpelling(enum CXCompletionChunkKind Kind) {
  switch (Kind) {
  case CXCompletionChunk_Optional: return "Optional";
  case CXCompletionChunk_TypedText: return "TypedText";
  case CXCompletionChunk_Text: return "Text";
  case CXCompletionChunk_Placeholder: return "Placeholder";
  case CXCompletionChunk_Informative: return "Informative";
  case CXCompletionChunk_CurrentParameter: return "CurrentParameter";
  case CXCompletionChunk_LeftParen: return "LeftParen";
  case CXCompletionChunk_RightParen: return "RightParen";
  case CXCompletionChunk_LeftBracket: return "LeftBracket";
  case CXCompletionChunk_RightBracket: return "RightBracket";
  case CXCompletionChunk_LeftBrace: return "LeftBrace";
  case CXCompletionChunk_RightBrace: return "RightBrace";
  case CXCompletionChunk_LeftAngle: return "LeftAngle";
  case CXCompletionChunk_RightAngle: return "RightAngle";
  case CXCompletionChunk_Comma: return "Comma";
  }
  
  return "Unknown";
}

void print_completion_string(CXCompletionString completion_string, FILE *file) {
  int I, N;
  
  N = clang_getNumCompletionChunks(completion_string);
  for (I = 0; I != N; ++I) {
    const char *text = 0;
    enum CXCompletionChunkKind Kind
      = clang_getCompletionChunkKind(completion_string, I);
    
    if (Kind == CXCompletionChunk_Optional) {
      fprintf(file, "{Optional ");
      print_completion_string(
                clang_getCompletionChunkCompletionString(completion_string, I), 
                              file);
      fprintf(file, "}");
      continue;
    }
    
    text = clang_getCompletionChunkText(completion_string, I);
    fprintf(file, "{%s %s}", 
            clang_getCompletionChunkKindSpelling(Kind),
            text? text : "");
  }
}

void print_completion_result(CXCompletionResult *completion_result,
                             CXClientData client_data) {
  FILE *file = (FILE *)client_data;
  fprintf(file, "%s:", 
          clang_getCursorKindSpelling(completion_result->CursorKind));
  print_completion_string(completion_result->CompletionString, file);
  fprintf(file, "\n");
}

int perform_code_completion(int argc, const char **argv) {
  const char *input = argv[1];
  char *filename = 0;
  unsigned line;
  unsigned column;
  CXIndex CIdx;
  int errorCode;

  input += strlen("-code-completion-at=");
  if ((errorCode = parse_file_line_column(input, &filename, &line, &column)))
    return errorCode;

  CIdx = clang_createIndex(0, 0);
  clang_codeComplete(CIdx, argv[argc - 1], argc - 3, argv + 2, 
                     filename, line, column, &print_completion_result, stdout);
  clang_disposeIndex(CIdx);
  free(filename);
  
  return 0;
}

/******************************************************************************/
/* Command line processing.                                                   */
/******************************************************************************/

static void print_usage(void) {
  fprintf(stderr,
    "usage: c-index-test -code-completion-at=<site> <compiler arguments>\n"
    "       c-index-test -test-load-tu <AST file> <symbol filter>\n\n"
    " <symbol filter> options for -test-load-tu:\n%s",
    "   all - load all symbols, including those from PCH\n"
    "   local - load all symbols except those in PCH\n"
    "   category - only load ObjC categories (non-PCH)\n"
    "   interface - only load ObjC interfaces (non-PCH)\n"
    "   protocol - only load ObjC protocols (non-PCH)\n"
    "   function - only load functions (non-PCH)\n"
    "   typedef - only load typdefs (non-PCH)\n\n");
}

int main(int argc, const char **argv) {
  if (argc > 2 && strstr(argv[1], "-code-completion-at=") == argv[1])
    return perform_code_completion(argc, argv);
  if (argc == 4 && strcmp(argv[1], "-test-load-tu") == 0)
    return perform_test_load_tu(argv[2], argv[3]);

  print_usage();
  return 1;
}
