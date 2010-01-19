/* c-index-test.c */

#include "clang-c/Index.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

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

static unsigned CreateTranslationUnit(CXIndex Idx, const char *file,
                                      CXTranslationUnit *TU) {
  
  *TU = clang_createTranslationUnit(Idx, file);
  if (!TU) {
    fprintf(stderr, "Unable to load translation unit from '%s'!\n", file);
    return 0;
  }  
  return 1;
}

/******************************************************************************/
/* Pretty-printing.                                                           */
/******************************************************************************/

static void PrintCursor(CXCursor Cursor) {
  if (clang_isInvalid(Cursor.kind))
    printf("Invalid Cursor => %s", clang_getCursorKindSpelling(Cursor.kind));
  else {
    CXString string;
    CXCursor Referenced;
    string = clang_getCursorSpelling(Cursor);
    printf("%s=%s", clang_getCursorKindSpelling(Cursor.kind),
                      clang_getCString(string));
    clang_disposeString(string);
    
    Referenced = clang_getCursorReferenced(Cursor);
    if (!clang_equalCursors(Referenced, clang_getNullCursor())) {
      CXSourceLocation Loc = clang_getCursorLocation(Referenced);
      printf(":%d:%d", Loc.line, Loc.column);
    }

    if (clang_isCursorDefinition(Cursor))
      printf(" (Definition)");
  }
}

static const char* GetCursorSource(CXCursor Cursor) {  
  const char *source = clang_getFileName(clang_getCursorLocation(Cursor).file);
  if (!source)
    return "<invalid loc>";  
  return basename(source);
}

/******************************************************************************/
/* Logic for testing clang_loadTranslationUnit().                             */
/******************************************************************************/

static const char *FileCheckPrefix = "CHECK";

static void PrintCursorExtent(CXCursor C) {
  CXSourceRange extent = clang_getCursorExtent(C);
  /* FIXME: Better way to check for empty extents? */
  if (!extent.begin.file)
    return;
  printf(" [Extent=%d:%d:%d:%d]", extent.begin.line, extent.begin.column,
         extent.end.line, extent.end.column);
}

static void DeclVisitor(CXDecl Dcl, CXCursor Cursor, CXClientData Filter) {
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    CXSourceLocation Loc = clang_getCursorLocation(Cursor);
    const char *source = clang_getFileName(Loc.file);
    if (!source)
      source = "<invalid loc>";  
    printf("// %s: %s:%d:%d: ", FileCheckPrefix, source, Loc.line, Loc.column);
    PrintCursor(Cursor);
    PrintCursorExtent(Cursor);

    printf("\n");
  }
}

static void TranslationUnitVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                                   CXClientData Filter) {
  if (!Filter || (Cursor.kind == *(enum CXCursorKind *)Filter)) {
    CXDecl D;
    CXSourceLocation Loc = clang_getCursorLocation(Cursor);
    printf("// %s: %s:%d:%d: ", FileCheckPrefix,
           GetCursorSource(Cursor), Loc.line, Loc.column);
    PrintCursor(Cursor);
    
    D = clang_getCursorDecl(Cursor);
    if (!D) {
      printf("\n");
      return;
    }
    
    PrintCursorExtent(Cursor);
    printf("\n");    
    clang_loadDeclaration(D, DeclVisitor, 0);
  }
}

static void FunctionScanVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                                CXClientData Filter) {
  const char *startBuf, *endBuf;
  unsigned startLine, startColumn, endLine, endColumn, curLine, curColumn;
  CXCursor Ref;

  if (Cursor.kind != CXCursor_FunctionDecl ||
      !clang_isCursorDefinition(Cursor))
    return;

  clang_getDefinitionSpellingAndExtent(Cursor, &startBuf, &endBuf,
                                       &startLine, &startColumn,
                                       &endLine, &endColumn);
  /* Probe the entire body, looking for both decls and refs. */
  curLine = startLine;
  curColumn = startColumn;

  while (startBuf < endBuf) {
    CXSourceLocation Loc;
    const char *source = 0;
    
    if (*startBuf == '\n') {
      startBuf++;
      curLine++;
      curColumn = 1;
    } else if (*startBuf != '\t')
      curColumn++;
          
    Loc = clang_getCursorLocation(Cursor);
    source = clang_getFileName(Loc.file);
    if (source) {
      Ref = clang_getCursor(Unit, source, curLine, curColumn);
      if (Ref.kind == CXCursor_NoDeclFound) {
        /* Nothing found here; that's fine. */
      } else if (Ref.kind != CXCursor_FunctionDecl) {
        printf("// %s: %s:%d:%d: ", FileCheckPrefix, GetCursorSource(Ref),
               curLine, curColumn);
        PrintCursor(Ref);
        printf("\n");
      }
    }
    startBuf++;
  }
}

/******************************************************************************/
/* USR testing.                                                               */
/******************************************************************************/

static void USRDeclVisitor(CXDecl D, CXCursor C, CXClientData Filter) {
  if (!Filter || (C.kind == *(enum CXCursorKind *)Filter)) {
    CXString USR = clang_getCursorUSR(C);
    if (!USR.Spelling) {
      clang_disposeString(USR);
      return;
    }
    printf("// %s: %s %s", FileCheckPrefix, GetCursorSource(C), USR.Spelling);
    PrintCursorExtent(C);
    printf("\n");
    clang_disposeString(USR);
  }
}

static void USRVisitor(CXTranslationUnit Unit, CXCursor Cursor,
                       CXClientData Filter) {
  CXDecl D = clang_getCursorDecl(Cursor);
  if (D) {
    /* USRDeclVisitor(Unit, Cursor.decl, Cursor, Filter);*/
    clang_loadDeclaration(D, USRDeclVisitor, 0);
  }
}

/******************************************************************************/
/* Loading ASTs/source.                                                       */
/******************************************************************************/

static int perform_test_load(CXIndex Idx, CXTranslationUnit TU,
                             const char *filter, const char *prefix,
                             CXTranslationUnitIterator Visitor) {
  enum CXCursorKind K = CXCursor_NotImplemented;
  enum CXCursorKind *ck = &K;

  if (prefix)
    FileCheckPrefix = prefix;  
  
  /* Perform some simple filtering. */
  if (!strcmp(filter, "all") || !strcmp(filter, "local")) ck = NULL;
  else if (!strcmp(filter, "category")) K = CXCursor_ObjCCategoryDecl;
  else if (!strcmp(filter, "interface")) K = CXCursor_ObjCInterfaceDecl;
  else if (!strcmp(filter, "protocol")) K = CXCursor_ObjCProtocolDecl;
  else if (!strcmp(filter, "function")) K = CXCursor_FunctionDecl;
  else if (!strcmp(filter, "typedef")) K = CXCursor_TypedefDecl;
  else if (!strcmp(filter, "scan-function")) Visitor = FunctionScanVisitor;
  else {
    fprintf(stderr, "Unknown filter for -test-load-tu: %s\n", filter);
    return 1;
  }
  
  clang_loadTranslationUnit(TU, Visitor, ck);
  clang_disposeTranslationUnit(TU);
  return 0;
}

int perform_test_load_tu(const char *file, const char *filter,
                         const char *prefix,
                         CXTranslationUnitIterator Visitor) {
  CXIndex Idx;
  CXTranslationUnit TU;
  Idx = clang_createIndex(/* excludeDeclsFromPCH */ 
                          !strcmp(filter, "local") ? 1 : 0, 
                          /* displayDiagnostics */ 1);
  
  if (!CreateTranslationUnit(Idx, file, &TU))
    return 1;

  return perform_test_load(Idx, TU, filter, prefix, Visitor);
}

int perform_test_load_source(int argc, const char **argv, const char *filter,
                             CXTranslationUnitIterator Visitor) {
  const char *UseExternalASTs =
    getenv("CINDEXTEST_USE_EXTERNAL_AST_GENERATION");
  CXIndex Idx;
  CXTranslationUnit TU;
  Idx = clang_createIndex(/* excludeDeclsFromPCH */
                          !strcmp(filter, "local") ? 1 : 0,
                          /* displayDiagnostics */ 1);

  if (UseExternalASTs && strlen(UseExternalASTs))
    clang_setUseExternalASTGeneration(Idx, 1);

  TU = clang_createTranslationUnitFromSourceFile(Idx, 0, argc, argv);
  if (!TU) {
    fprintf(stderr, "Unable to load translation unit!\n");
    return 1;
  }

  return perform_test_load(Idx, TU, filter, NULL, Visitor);
}

/******************************************************************************/
/* Logic for testing clang_getCursor().                                       */
/******************************************************************************/

static void print_cursor_file_scan(CXCursor cursor,
                                   unsigned start_line, unsigned start_col,
                                   unsigned end_line, unsigned end_col,
                                   const char *prefix) {
  printf("// %s: ", FileCheckPrefix);
  if (prefix)
    printf("-%s", prefix);
  printf("{start_line=%d start_col=%d end_line=%d end_col=%d} ",
          start_line, start_col, end_line, end_col);
  PrintCursor(cursor);
  printf("\n");
}

static int perform_file_scan(const char *ast_file, const char *source_file,
                             const char *prefix) {
  CXIndex Idx;
  CXTranslationUnit TU;
  FILE *fp;
  unsigned line;
  CXCursor prevCursor;
  unsigned printed;
  unsigned start_line, start_col, last_line, last_col;
  size_t i;
  
  if (!(Idx = clang_createIndex(/* excludeDeclsFromPCH */ 1,
                                /* displayDiagnostics */ 1))) {
    fprintf(stderr, "Could not create Index\n");
    return 1;
  }
  
  if (!CreateTranslationUnit(Idx, ast_file, &TU))
    return 1;
  
  if ((fp = fopen(source_file, "r")) == NULL) {
    fprintf(stderr, "Could not open '%s'\n", source_file);
    return 1;
  }
  
  line = 0;
  prevCursor = clang_getNullCursor();
  printed = 0;
  start_line = last_line = 1;
  start_col = last_col = 1;
  
  while (!feof(fp)) {
    size_t len = 0;
    int c;

    while ((c = fgetc(fp)) != EOF) {
      len++;
      if (c == '\n')
        break;
    }

    ++line;
    
    for (i = 0; i < len ; ++i) {
      CXCursor cursor;
      cursor = clang_getCursor(TU, source_file, line, i+1);

      if (!clang_equalCursors(cursor, prevCursor) &&
          prevCursor.kind != CXCursor_InvalidFile) {
        print_cursor_file_scan(prevCursor, start_line, start_col,
                               last_line, last_col, prefix);
        printed = 1;
        start_line = line;
        start_col = (unsigned) i+1;
      }
      else {
        printed = 0;
      }
      
      prevCursor = cursor;
      last_line = line;
      last_col = (unsigned) i+1;
    }    
  }
  
  if (!printed && prevCursor.kind != CXCursor_InvalidFile) {
    print_cursor_file_scan(prevCursor, start_line, start_col,
                           last_line, last_col, prefix);
  }  
  
  fclose(fp);
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
  case CXCompletionChunk_ResultType: return "ResultType";
  case CXCompletionChunk_Colon: return "Colon";
  case CXCompletionChunk_SemiColon: return "SemiColon";
  case CXCompletionChunk_Equal: return "Equal";
  case CXCompletionChunk_HorizontalSpace: return "HorizontalSpace";
  case CXCompletionChunk_VerticalSpace: return "VerticalSpace";
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

void free_remapped_files(struct CXUnsavedFile *unsaved_files,
                         int num_unsaved_files) {
  int i;
  for (i = 0; i != num_unsaved_files; ++i) {
    free((char *)unsaved_files[i].Filename);
    free((char *)unsaved_files[i].Contents);
  }
}

int parse_remapped_files(int argc, const char **argv, int start_arg,
                         struct CXUnsavedFile **unsaved_files,
                          int *num_unsaved_files) {
  int i;
  int arg;
  int prefix_len = strlen("-remap-file=");
  *unsaved_files = 0;
  *num_unsaved_files = 0;

  /* Count the number of remapped files. */
  for (arg = start_arg; arg < argc; ++arg) {
    if (strncmp(argv[arg], "-remap-file=", prefix_len))
      break;

    ++*num_unsaved_files;
  }

  if (*num_unsaved_files == 0)
    return 0;

  *unsaved_files
    = (struct CXUnsavedFile *)malloc(sizeof(struct CXUnsavedFile) * 
                                     *num_unsaved_files);
  for (arg = start_arg, i = 0; i != *num_unsaved_files; ++i, ++arg) {
    struct CXUnsavedFile *unsaved = *unsaved_files + i;
    const char *arg_string = argv[arg] + prefix_len;
    int filename_len;
    char *filename;
    char *contents;
    FILE *to_file;
    const char *semi = strchr(arg_string, ';');
    if (!semi) {
      fprintf(stderr, 
              "error: -remap-file=from;to argument is missing semicolon\n");
      free_remapped_files(*unsaved_files, i);
      *unsaved_files = 0;
      *num_unsaved_files = 0;
      return -1;
    }

    /* Open the file that we're remapping to. */
    to_file = fopen(semi + 1, "r");
    if (!to_file) {
      fprintf(stderr, "error: cannot open file %s that we are remapping to\n",
              semi + 1);
      free_remapped_files(*unsaved_files, i);
      *unsaved_files = 0;
      *num_unsaved_files = 0;
      return -1;
    }

    /* Determine the length of the file we're remapping to. */
    fseek(to_file, 0, SEEK_END);
    unsaved->Length = ftell(to_file);
    fseek(to_file, 0, SEEK_SET);
    
    /* Read the contents of the file we're remapping to. */
    contents = (char *)malloc(unsaved->Length + 1);
    if (fread(contents, 1, unsaved->Length, to_file) != unsaved->Length) {
      fprintf(stderr, "error: unexpected %s reading 'to' file %s\n",
              (feof(to_file) ? "EOF" : "error"), semi + 1);
      fclose(to_file);
      free_remapped_files(*unsaved_files, i);
      *unsaved_files = 0;
      *num_unsaved_files = 0;
      return -1;
    }
    contents[unsaved->Length] = 0;
    unsaved->Contents = contents;

    /* Close the file. */
    fclose(to_file);
    
    /* Copy the file name that we're remapping from. */
    filename_len = semi - arg_string;
    filename = (char *)malloc(filename_len + 1);
    memcpy(filename, arg_string, filename_len);
    filename[filename_len] = 0;
    unsaved->Filename = filename;
  }

  return 0;
}

int perform_code_completion(int argc, const char **argv) {
  const char *input = argv[1];
  char *filename = 0;
  unsigned line;
  unsigned column;
  CXIndex CIdx;
  int errorCode;
  struct CXUnsavedFile *unsaved_files = 0;
  int num_unsaved_files = 0;
  CXCodeCompleteResults *results = 0;

  input += strlen("-code-completion-at=");
  if ((errorCode = parse_file_line_column(input, &filename, &line, &column)))
    return errorCode;

  if (parse_remapped_files(argc, argv, 2, &unsaved_files, &num_unsaved_files))
    return -1;

  CIdx = clang_createIndex(0, 0);
  results = clang_codeComplete(CIdx, 
                               argv[argc - 1], argc - num_unsaved_files - 3, 
                               argv + num_unsaved_files + 2, 
                               num_unsaved_files, unsaved_files,
                               filename, line, column);
  if (results) {
    unsigned i, n = results->NumResults;
    for (i = 0; i != n; ++i)
      print_completion_result(results->Results + i, stdout);
    clang_disposeCodeCompleteResults(results);
  }

  clang_disposeIndex(CIdx);
  free(filename);
  
  free_remapped_files(unsaved_files, num_unsaved_files);

  return 0;
}

typedef struct {
  char *filename;
  unsigned line;
  unsigned column;
} CursorSourceLocation;

int inspect_cursor_at(int argc, const char **argv) {
  CXIndex CIdx;
  int errorCode;
  struct CXUnsavedFile *unsaved_files = 0;
  int num_unsaved_files = 0;
  CXTranslationUnit TU;
  CXCursor Cursor;
  CursorSourceLocation *Locations = 0;
  unsigned NumLocations = 0, Loc;
  
  /* Count the number of locations. */ 
  while (strstr(argv[NumLocations+1], "-cursor-at=") == argv[NumLocations+1])
    ++NumLocations;
  
  /* Parse the locations. */
  assert(NumLocations > 0 && "Unable to count locations?");
  Locations = (CursorSourceLocation *)malloc(
                                  NumLocations * sizeof(CursorSourceLocation));
  for (Loc = 0; Loc < NumLocations; ++Loc) {
    const char *input = argv[Loc + 1] + strlen("-cursor-at=");
    if ((errorCode = parse_file_line_column(input, &Locations[Loc].filename, 
                                            &Locations[Loc].line, 
                                            &Locations[Loc].column)))
      return errorCode;
  }
  
  if (parse_remapped_files(argc, argv, NumLocations + 1, &unsaved_files, 
                           &num_unsaved_files))
    return -1;
  
  if (num_unsaved_files > 0) {
    fprintf(stderr, "cannot remap files when looking for a cursor\n");
    return -1;
  }
  
  CIdx = clang_createIndex(0, 1);
  TU = clang_createTranslationUnitFromSourceFile(CIdx, argv[argc - 1],
                                  argc - num_unsaved_files - 2 - NumLocations,
                                   argv + num_unsaved_files + 1 + NumLocations);
  if (!TU) {
    fprintf(stderr, "unable to parse input\n");
    return -1;
  }
  
  for (Loc = 0; Loc < NumLocations; ++Loc) {
    Cursor = clang_getCursor(TU, Locations[Loc].filename, 
                             Locations[Loc].line, Locations[Loc].column);  
    PrintCursor(Cursor);
    printf("\n");
    free(Locations[Loc].filename);
  }
  
  clang_disposeTranslationUnit(TU);
  clang_disposeIndex(CIdx);
  free(Locations);
  free_remapped_files(unsaved_files, num_unsaved_files);
  return 0;
}

/******************************************************************************/
/* Command line processing.                                                   */
/******************************************************************************/

static CXTranslationUnitIterator GetVisitor(const char *s) {
  if (s[0] == '\0')
    return TranslationUnitVisitor;
  if (strcmp(s, "-usrs") == 0)
    return USRVisitor;
  return NULL;
}

static void print_usage(void) {
  fprintf(stderr,
    "usage: c-index-test -code-completion-at=<site> <compiler arguments>\n"
    "       c-index-test -cursor-at=<site> <compiler arguments>\n"
    "       c-index-test -test-file-scan <AST file> <source file> "
          "[FileCheck prefix]\n"
    "       c-index-test -test-load-tu <AST file> <symbol filter> "
          "[FileCheck prefix]\n"
    "       c-index-test -test-load-tu-usrs <AST file> <symbol filter> "
           "[FileCheck prefix]\n"
    "       c-index-test -test-load-source <symbol filter> {<args>}*\n"
    "       c-index-test -test-load-source-usrs <symbol filter> {<args>}*\n\n");
  fprintf(stderr,
    " <symbol filter> values:\n%s",
    "   all - load all symbols, including those from PCH\n"
    "   local - load all symbols except those in PCH\n"
    "   category - only load ObjC categories (non-PCH)\n"
    "   interface - only load ObjC interfaces (non-PCH)\n"
    "   protocol - only load ObjC protocols (non-PCH)\n"
    "   function - only load functions (non-PCH)\n"
    "   typedef - only load typdefs (non-PCH)\n"
    "   scan-function - scan function bodies (non-PCH)\n\n");
}

int main(int argc, const char **argv) {
  if (argc > 2 && strstr(argv[1], "-code-completion-at=") == argv[1])
    return perform_code_completion(argc, argv);
  if (argc > 2 && strstr(argv[1], "-cursor-at=") == argv[1])
    return inspect_cursor_at(argc, argv);
  else if (argc >= 4 && strncmp(argv[1], "-test-load-tu", 13) == 0) {
    CXTranslationUnitIterator I = GetVisitor(argv[1] + 13);
    if (I)
      return perform_test_load_tu(argv[2], argv[3], argc >= 5 ? argv[4] : 0, I);
  }
  else if (argc >= 4 && strncmp(argv[1], "-test-load-source", 17) == 0) {
    CXTranslationUnitIterator I = GetVisitor(argv[1] + 17);
    if (I)
      return perform_test_load_source(argc - 3, argv + 3, argv[2], I);
  }
  else if (argc >= 4 && strcmp(argv[1], "-test-file-scan") == 0)
    return perform_file_scan(argv[2], argv[3],
                             argc >= 5 ? argv[4] : 0);

  print_usage();
  return 1;
}
