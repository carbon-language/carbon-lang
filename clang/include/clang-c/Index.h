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

#ifdef __cplusplus
extern "C" {
#endif

/*
   Clang indeX abstractions. The backing store for the following API's will be 
   clangs PCH file (which contains AST's, or Abstract Syntax Trees). PCH files
   are created by the following command:
   
   "clang -S -Xclang -emit-pch <sourcefile.langsuffix> -o <sourcefile.ast>". 
   
   If the ast file format ends up diverging from the pch file format, we will 
   need to add a new switch (-emit-ast). For now, the contents are identical.

   Naming Conventions: To avoid namespace pollution, data types are prefixed 
   with "CX" and functions are prefixed with "clang_".
*/
typedef void *CXIndex;            /* An indexing instance. */

typedef void *CXTranslationUnit;  /* A translation unit instance. */

typedef void *CXDecl;    /* A specific declaration within a translation unit. */

/* Cursors represent declarations and references (provides line/column info). */
enum CXCursorKind {
 CXCursor_Invalid                       = 0,
 
 /* Declarations */
 CXCursor_FirstDecl                     = 1,
 CXCursor_TypedefDecl                   = 1,
 CXCursor_EnumDecl                      = 2,
 CXCursor_EnumConstantDecl              = 3,
 CXCursor_RecordDecl                    = 4, /* struct/union/class */
 CXCursor_FieldDecl                     = 5,
 CXCursor_FunctionDecl                  = 6,
 CXCursor_VarDecl                       = 7,
 CXCursor_ParmDecl                      = 8,
 CXCursor_ObjCInterfaceDecl             = 9,
 CXCursor_ObjCCategoryDecl              = 10,
 CXCursor_ObjCProtocolDecl              = 11,
 CXCursor_ObjCPropertyDecl              = 12,
 CXCursor_ObjCIvarDecl                  = 13,
 CXCursor_ObjCMethodDecl                = 14,
 CXCursor_LastDecl                      = 14,
 
 /* References */
 CXCursor_FirstRef                      = 19,
 CXCursor_ObjCClassRef                  = 19,            
 CXCursor_ObjCProtocolRef               = 20,
 CXCursor_ObjCMessageRef                = 21,
 CXCursor_ObjCSelectorRef               = 22,
 CXCursor_LastRef                       = 23
};

/* A cursor into the CXTranslationUnit. */
typedef struct {
  enum CXCursorKind kind;
  CXDecl decl;
  
  /* FIXME: Handle references. */
} CXCursor;  

/* A unique token for looking up "visible" CXDecls from a CXTranslationUnit. */
typedef void *CXEntity;     

CXIndex clang_createIndex();

CXTranslationUnit clang_createTranslationUnit(
  CXIndex, const char *ast_filename
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
typedef void (*CXTranslationUnitIterator)(CXTranslationUnit, CXCursor);
void clang_loadTranslationUnit(CXTranslationUnit, CXTranslationUnitIterator);

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
typedef void (*CXDeclIterator)(CXTranslationUnit, CXDecl, void *clientData);

void clang_loadDeclaration(CXDecl, CXDeclIterator);

/*
 * CXEntity Operations.
 */
const char *clang_getDeclarationName(CXEntity);
const char *clang_getURI(CXEntity);
CXEntity clang_getEntity(const char *URI);
/*
 * CXDecl Operations.
 */
CXCursor clang_getCursorFromDecl(CXDecl);
CXEntity clang_getEntityFromDecl(CXDecl);
const char *clang_getDeclSpelling(CXDecl);
/*
 * CXCursor Operations.
 */
CXCursor clang_getCursor(CXTranslationUnit, const char *source_name, 
                         unsigned line, unsigned column);

enum CXCursorKind clang_getCursorKind(CXCursor);
unsigned clang_isDeclaration(enum CXCursorKind);

unsigned clang_getCursorLine(CXCursor);
unsigned clang_getCursorColumn(CXCursor);
const char *clang_getCursorSource(CXCursor);
const char *clang_getKindSpelling(enum CXCursorKind Kind);

/*
 * If CXCursorKind == Cursor_Reference, then this will return the referenced declaration.
 * If CXCursorKind == Cursor_Declaration, then this will return the declaration.
 */
CXDecl clang_getCursorDecl(CXCursor);

#ifdef __cplusplus
}
#endif
#endif

