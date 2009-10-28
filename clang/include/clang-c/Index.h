/*===-- clang-c/Index.h - Indexing Public C Interface -------------*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides a public inferface to a Clang library for extracting  *|
|* high-level symbol information from source files without exposing the full  *|
|* Clang C++ API.                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef CLANG_C_INDEX_H
#define CLANG_C_INDEX_H

#include <sys/stat.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MSVC DLL import/export. */
#ifdef _MSC_VER
  #ifdef _CINDEX_LIB_
    #define CINDEX_LINKAGE __declspec(dllexport)
  #else
    #define CINDEX_LINKAGE __declspec(dllimport)
  #endif
#else
  #define CINDEX_LINKAGE
#endif

/*
   Clang indeX abstractions. The backing store for the following API's will be 
   clangs AST file (currently based on PCH). AST files are created as follows:
   
   "clang -emit-ast <sourcefile.langsuffix> -o <sourcefile.ast>". 
   
   Naming Conventions: To avoid namespace pollution, data types are prefixed 
   with "CX" and functions are prefixed with "clang_".
*/
typedef void *CXIndex;            /* An indexing instance. */

typedef void *CXTranslationUnit;  /* A translation unit instance. */

typedef void *CXFile;    /* A source file */
typedef void *CXDecl;    /* A specific declaration within a translation unit. */
typedef void *CXStmt;    /* A specific statement within a function/method */

/* Cursors represent declarations, definitions, and references. */
enum CXCursorKind {
 /* Declarations */
 CXCursor_FirstDecl                     = 1,
 CXCursor_TypedefDecl                   = 2,
 CXCursor_StructDecl                    = 3, 
 CXCursor_UnionDecl                     = 4,
 CXCursor_ClassDecl                     = 5,
 CXCursor_EnumDecl                      = 6,
 CXCursor_FieldDecl                     = 7,
 CXCursor_EnumConstantDecl              = 8,
 CXCursor_FunctionDecl                  = 9,
 CXCursor_VarDecl                       = 10,
 CXCursor_ParmDecl                      = 11,
 CXCursor_ObjCInterfaceDecl             = 12,
 CXCursor_ObjCCategoryDecl              = 13,
 CXCursor_ObjCProtocolDecl              = 14,
 CXCursor_ObjCPropertyDecl              = 15,
 CXCursor_ObjCIvarDecl                  = 16,
 CXCursor_ObjCInstanceMethodDecl        = 17,
 CXCursor_ObjCClassMethodDecl           = 18,
 CXCursor_LastDecl                      = 18,
 
 /* Definitions */
 CXCursor_FirstDefn                     = 32,
 CXCursor_FunctionDefn                  = 32,
 CXCursor_ObjCClassDefn                 = 33,
 CXCursor_ObjCCategoryDefn              = 34,
 CXCursor_ObjCInstanceMethodDefn        = 35,
 CXCursor_ObjCClassMethodDefn           = 36,
 CXCursor_LastDefn                      = 36,
   
 /* References */
 CXCursor_FirstRef                      = 40, /* Decl references */
 CXCursor_ObjCSuperClassRef             = 40,            
 CXCursor_ObjCProtocolRef               = 41,
 CXCursor_ObjCClassRef                  = 42,
 
 CXCursor_ObjCSelectorRef               = 43, /* Expression references */
 CXCursor_ObjCIvarRef                   = 44,
 CXCursor_VarRef                        = 45,
 CXCursor_FunctionRef                   = 46,
 CXCursor_EnumConstantRef               = 47,
 CXCursor_MemberRef                     = 48,
 CXCursor_LastRef                       = 48,
 
 /* Error conditions */
 CXCursor_FirstInvalid                  = 70,
 CXCursor_InvalidFile                   = 70,
 CXCursor_NoDeclFound                   = 71,
 CXCursor_NotImplemented                = 72,
 CXCursor_LastInvalid                   = 72
};

/* A cursor into the CXTranslationUnit. */

typedef struct {
  enum CXCursorKind kind;
  CXDecl decl;
  CXStmt stmt; /* expression reference */
} CXCursor;  

/* A unique token for looking up "visible" CXDecls from a CXTranslationUnit. */
typedef void *CXEntity;

/**  
 * \brief clang_createIndex() provides a shared context for creating
 * translation units. It provides two options:
 *
 * - excludeDeclarationsFromPCH: When non-zero, allows enumeration of "local"
 * declarations (when loading any new translation units). A "local" declaration
 * is one that belongs in the translation unit itself and not in a precompiled 
 * header that was used by the translation unit. If zero, all declarations
 * will be enumerated.
 *
 * - displayDiagnostics: when non-zero, diagnostics will be output. If zero,
 * diagnostics will be ignored.
 *
 * Here is an example:
 *
 *   // excludeDeclsFromPCH = 1, displayDiagnostics = 1
 *   Idx = clang_createIndex(1, 1);
 *
 *   // IndexTest.pch was produced with the following command:
 *   // "clang -x c IndexTest.h -emit-ast -o IndexTest.pch"
 *   TU = clang_createTranslationUnit(Idx, "IndexTest.pch");
 *
 *   // This will load all the symbols from 'IndexTest.pch'
 *   clang_loadTranslationUnit(TU, TranslationUnitVisitor, 0);
 *   clang_disposeTranslationUnit(TU);
 *
 *   // This will load all the symbols from 'IndexTest.c', excluding symbols
 *   // from 'IndexTest.pch'.
 *   char *args[] = { "-Xclang", "-include-pch=IndexTest.pch", 0 };
 *   TU = clang_createTranslationUnitFromSourceFile(Idx, "IndexTest.c", 2, args);
 *   clang_loadTranslationUnit(TU, TranslationUnitVisitor, 0);
 *   clang_disposeTranslationUnit(TU);
 *
 * This process of creating the 'pch', loading it separately, and using it (via
 * -include-pch) allows 'excludeDeclsFromPCH' to remove redundant callbacks
 * (which gives the indexer the same performance benefit as the compiler).
 */
CINDEX_LINKAGE CXIndex clang_createIndex(int excludeDeclarationsFromPCH,
                          int displayDiagnostics);
CINDEX_LINKAGE void clang_disposeIndex(CXIndex);

CINDEX_LINKAGE const char *clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit);

/* 
 * \brief Create a translation unit from an AST file (-emit-ast).
 */
CINDEX_LINKAGE CXTranslationUnit clang_createTranslationUnit(
  CXIndex, const char *ast_filename
);
/**
 * \brief Destroy the specified CXTranslationUnit object.
 */ 
CINDEX_LINKAGE void clang_disposeTranslationUnit(CXTranslationUnit);

/**
 * \brief Return the CXTranslationUnit for a given source file and the provided
 * command line arguments one would pass to the compiler.
 *
 * Note: The 'source_filename' argument is optional.  If the caller provides a NULL pointer,
 *  the name of the source file is expected to reside in the specified command line arguments.
 *
 * Note: When encountered in 'clang_command_line_args', the following options are ignored:
 *
 *   '-c'
 *   '-emit-ast'
 *   '-fsyntax-only'
 *   '-o <output file>'  (both '-o' and '<output file>' are ignored)
 *
 */
CINDEX_LINKAGE CXTranslationUnit clang_createTranslationUnitFromSourceFile(
  CXIndex CIdx, 
  const char *source_filename /* specify NULL if the source file is in clang_command_line_args */,
  int num_clang_command_line_args, 
  const char **clang_command_line_args
);

/*
   Usage: clang_loadTranslationUnit(). Will load the toplevel declarations
   within a translation unit, issuing a 'callback' for each one.

   void printObjCInterfaceNames(CXTranslationUnit X, CXCursor C) {
     if (clang_getCursorKind(C) == Cursor_Declaration) {
       CXDecl D = clang_getCursorDecl(C);
       if (clang_getDeclKind(D) == CXDecl_ObjC_interface)
         printf("@interface %s in file %s on line %d column %d\n",
                clang_getDeclSpelling(D), clang_getCursorSource(C),
                clang_getCursorLine(C), clang_getCursorColumn(C));
     }
   }
   static void usage {
     clang_loadTranslationUnit(CXTranslationUnit, printObjCInterfaceNames);
   }
*/
typedef void *CXClientData;
typedef void (*CXTranslationUnitIterator)(CXTranslationUnit, CXCursor, 
                                          CXClientData);
CINDEX_LINKAGE void clang_loadTranslationUnit(CXTranslationUnit, CXTranslationUnitIterator,
                               CXClientData);

/*
   Usage: clang_loadDeclaration(). Will load the declaration, issuing a 
   'callback' for each declaration/reference within the respective declaration.
   
   For interface declarations, this will index the super class, protocols, 
   ivars, methods, etc. For structure declarations, this will index the fields.
   For functions, this will index the parameters (and body, for function 
   definitions), local declarations/references.

   void getInterfaceDetails(CXDecl X, CXCursor C) {
     switch (clang_getCursorKind(C)) {
       case Cursor_ObjC_ClassRef:
         CXDecl SuperClass = clang_getCursorDecl(C);
       case Cursor_ObjC_ProtocolRef:
         CXDecl AdoptsProtocol = clang_getCursorDecl(C);
       case Cursor_Declaration:
         CXDecl AnIvarOrMethod = clang_getCursorDecl(C);
     }
   }
   static void usage() {
     if (clang_getDeclKind(D) == CXDecl_ObjC_interface) {
       clang_loadDeclaration(D, getInterfaceDetails);
     }
   }
*/
typedef void (*CXDeclIterator)(CXDecl, CXCursor, CXClientData);

CINDEX_LINKAGE void clang_loadDeclaration(CXDecl, CXDeclIterator, CXClientData);

/*
 * CXFile Operations.
 */
const char *clang_getFileName(CXFile SFile);
time_t clang_getFileTime(CXFile SFile);

/*
 * CXEntity Operations.
 */
CINDEX_LINKAGE const char *clang_getDeclarationName(CXEntity);
CINDEX_LINKAGE const char *clang_getURI(CXEntity);
CINDEX_LINKAGE CXEntity clang_getEntity(const char *URI);
/*
 * CXDecl Operations.
 */
CINDEX_LINKAGE CXCursor clang_getCursorFromDecl(CXDecl);
CINDEX_LINKAGE CXEntity clang_getEntityFromDecl(CXDecl);
CINDEX_LINKAGE const char *clang_getDeclSpelling(CXDecl);
CINDEX_LINKAGE unsigned clang_getDeclLine(CXDecl);
CINDEX_LINKAGE unsigned clang_getDeclColumn(CXDecl);
CINDEX_LINKAGE const char *clang_getDeclSource(CXDecl); /* deprecate */
CINDEX_LINKAGE CXFile clang_getDeclSourceFile(CXDecl);

/*
 * CXCursor Operations.
 */
/**
   Usage: clang_getCursor() will translate a source/line/column position
   into an AST cursor (to derive semantic information from the source code).
 */
CINDEX_LINKAGE CXCursor clang_getCursor(CXTranslationUnit, const char *source_name, 
                         unsigned line, unsigned column);

CINDEX_LINKAGE enum CXCursorKind clang_getCursorKind(CXCursor);
CINDEX_LINKAGE unsigned clang_isDeclaration(enum CXCursorKind);
CINDEX_LINKAGE unsigned clang_isReference(enum CXCursorKind);
CINDEX_LINKAGE unsigned clang_isDefinition(enum CXCursorKind);
CINDEX_LINKAGE unsigned clang_isInvalid(enum CXCursorKind);

CINDEX_LINKAGE unsigned clang_getCursorLine(CXCursor);
CINDEX_LINKAGE unsigned clang_getCursorColumn(CXCursor);
CINDEX_LINKAGE const char *clang_getCursorSpelling(CXCursor);
CINDEX_LINKAGE const char *clang_getCursorSource(CXCursor); /* deprecate */
CINDEX_LINKAGE CXFile clang_getCursorSourceFile(CXCursor);

/* for debug/testing */
CINDEX_LINKAGE const char *clang_getCursorKindSpelling(enum CXCursorKind Kind); 
CINDEX_LINKAGE void clang_getDefinitionSpellingAndExtent(CXCursor, 
                                          const char **startBuf, 
                                          const char **endBuf,
                                          unsigned *startLine,
                                          unsigned *startColumn,
                                          unsigned *endLine,
                                          unsigned *endColumn);

/*
 * If CXCursorKind == Cursor_Reference, then this will return the referenced
 * declaration.
 * If CXCursorKind == Cursor_Declaration, then this will return the declaration.
 */
CINDEX_LINKAGE CXDecl clang_getCursorDecl(CXCursor);

#ifdef __cplusplus
}
#endif
#endif

