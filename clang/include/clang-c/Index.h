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
   
   "clang -emit-pch <sourcefile.langsuffix> -o <sourcefile.ast>". 
   
   If the ast file format ends up diverging from the pch file format, we will 
   need to add a new switch (-emit-ast). For now, the contents are identical.

   Naming Conventions: To avoid namespace pollution, data types are prefixed 
   with "CX" and functions are prefixed with "clang_".
*/
typedef void *CXIndex;            /* An indexing instance. */

typedef void *CXTranslationUnit;  /* A translation unit instance. */

typedef void *CXCursor;  /* An opaque cursor into the CXTranslationUnit. */

/* Cursors represent declarations and references (provides line/column info). */
enum CXCursorKind {  
 CXCursor_Declaration,
 CXCursor_Reference,
 CXCursor_ObjC_ClassRef,
 CXCursor_ObjC_ProtocolRef,
 CXCursor_ObjC_MessageRef,
 CXCursor_ObjC_SelectorRef
};

typedef void *CXDecl;    /* A specific declaration within a translation unit. */

enum CXDeclKind {  /* The various kinds of declarations. */
 CXDecl_any,
 CXDecl_typedef,
 CXDecl_enum,
 CXDecl_enum_constant,
 CXDecl_record,
 CXDecl_field,
 CXDecl_function,
 CXDecl_variable,
 CXDecl_parameter,
 CXDecl_ObjC_interface,
 CXDecl_ObjC_category,
 CXDecl_ObjC_protocol,
 CXDecl_ObjC_property,
 CXDecl_ObjC_instance_variable,
 CXDecl_ObjC_instance_method,
 CXDecl_ObjC_class_method,
 CXDecl_ObjC_category_implementation,
 CXDecl_ObjC_class_implementation,
 CXDecl_ObjC_property_implementation
};

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
void clang_loadTranslationUnit(
  CXTranslationUnit, void (*callback)(CXTranslationUnit, CXCursor)
);

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
void clang_loadDeclaration(CXDecl, void (*callback)(CXDecl, CXCursor));

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
enum CXDeclKind clang_getDeclKind(CXDecl);
const char *clang_getDeclSpelling(CXDecl);
/*
 * CXCursor Operations.
 */
CXCursor clang_getCursor(CXTranslationUnit, const char *source_name, 
                         unsigned line, unsigned column);

enum CXCursorKind clang_getCursorKind(CXCursor);

unsigned clang_getCursorLine(CXCursor);
unsigned clang_getCursorColumn(CXCursor);
const char *clang_getCursorSource(CXCursor);

/*
 * If CXCursorKind == Cursor_Reference, then this will return the referenced declaration.
 * If CXCursorKind == Cursor_Declaration, then this will return the declaration.
 */
CXDecl clang_getCursorDecl(CXCursor);

#ifdef __cplusplus
}
#endif
#endif

