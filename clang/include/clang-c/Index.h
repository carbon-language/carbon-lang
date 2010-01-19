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
#include <time.h>

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
 CXCursor_TypedefDecl                   = 1,
 CXCursor_StructDecl                    = 2, 
 CXCursor_UnionDecl                     = 3,
 CXCursor_ClassDecl                     = 4,
 CXCursor_EnumDecl                      = 5,
 CXCursor_FieldDecl                     = 6,
 CXCursor_EnumConstantDecl              = 7,
 CXCursor_FunctionDecl                  = 8,
 CXCursor_VarDecl                       = 9,
 CXCursor_ParmDecl                      = 10,
 CXCursor_ObjCInterfaceDecl             = 11,
 CXCursor_ObjCCategoryDecl              = 12,
 CXCursor_ObjCProtocolDecl              = 13,
 CXCursor_ObjCPropertyDecl              = 14,
 CXCursor_ObjCIvarDecl                  = 15,
 CXCursor_ObjCInstanceMethodDecl        = 16,
 CXCursor_ObjCClassMethodDecl           = 17,
 CXCursor_ObjCImplementationDecl        = 18,
 CXCursor_ObjCCategoryImplDecl          = 19,
 CXCursor_LastDecl                      = 19,
 
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

/**
 * \brief Provides the contents of a file that has not yet been saved to disk.
 *
 * Each CXUnsavedFile instance provides the name of a file on the
 * system along with the current contents of that file that have not
 * yet been saved to disk.
 */
struct CXUnsavedFile {
  /** 
   * \brief The file whose contents have not yet been saved. 
   *
   * This file must already exist in the file system.
   */
  const char *Filename;

  /** 
   * \brief A null-terminated buffer containing the unsaved contents
   * of this file.
   */
  const char *Contents;

  /**
   * \brief The length of the unsaved contents of this buffer, not
   * counting the NULL at the end of the buffer.
   */
  unsigned long Length;
};

/* A cursor into the CXTranslationUnit. */

typedef struct {
  enum CXCursorKind kind;
  void *data[3];
} CXCursor;  

/* A unique token for looking up "visible" CXDecls from a CXTranslationUnit. */
typedef struct {
  CXIndex index;
  void *data;
} CXEntity;

/**
 * For functions returning a string that might or might not need
 * to be internally allocated and freed.
 * Use clang_getCString to access the C string value.
 * Use clang_disposeString to free the value.
 * Treat it as an opaque type.
 */
typedef struct {
  const char *Spelling;
  /* A 1 value indicates the clang_ indexing API needed to allocate the string
     (and it must be freed by clang_disposeString()). */
  int MustFreeString;
} CXString;

/* Get C string pointer from a CXString. */
CINDEX_LINKAGE const char *clang_getCString(CXString string);

/* Free CXString. */
CINDEX_LINKAGE void clang_disposeString(CXString string);

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
CINDEX_LINKAGE void clang_disposeIndex(CXIndex index);
CINDEX_LINKAGE CXString
clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit);

/* 
 * \brief Request that AST's be generated external for API calls which parse
 * source code on the fly, e.g. \see createTranslationUnitFromSourceFile.
 *
 * Note: This is for debugging purposes only, and may be removed at a later
 * date.
 *
 * \param index - The index to update.
 * \param value - The new flag value.
 */
CINDEX_LINKAGE void clang_setUseExternalASTGeneration(CXIndex index,
                                                      int value);

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
 * Note: The 'source_filename' argument is optional.  If the caller provides a
 * NULL pointer, the name of the source file is expected to reside in the
 * specified command line arguments.
 *
 * Note: When encountered in 'clang_command_line_args', the following options
 * are ignored:
 *
 *   '-c'
 *   '-emit-ast'
 *   '-fsyntax-only'
 *   '-o <output file>'  (both '-o' and '<output file>' are ignored)
 *
 *
 * \param source_filename - The name of the source file to load, or NULL if the
 * source file is included in clang_command_line_args.
 */
CINDEX_LINKAGE CXTranslationUnit clang_createTranslationUnitFromSourceFile(
  CXIndex CIdx,
  const char *source_filename,
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
CINDEX_LINKAGE void clang_loadTranslationUnit(CXTranslationUnit,
                                              CXTranslationUnitIterator,
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
CINDEX_LINKAGE const char *clang_getFileName(CXFile SFile);
CINDEX_LINKAGE time_t clang_getFileTime(CXFile SFile);

/*
 * CXEntity Operations.
 */
  
/* clang_getDeclaration() maps from a CXEntity to the matching CXDecl (if any)
 *  in a specified translation unit. */
CINDEX_LINKAGE CXDecl clang_getDeclaration(CXEntity, CXTranslationUnit);

/*
 * CXDecl Operations.
 */
CINDEX_LINKAGE CXCursor clang_getCursorFromDecl(CXDecl);
CINDEX_LINKAGE CXEntity clang_getEntityFromDecl(CXIndex, CXDecl);
CINDEX_LINKAGE CXString clang_getDeclSpelling(CXDecl);
CINDEX_LINKAGE unsigned clang_getDeclLine(CXDecl); /* deprecate */
CINDEX_LINKAGE unsigned clang_getDeclColumn(CXDecl); /* deprecate */
CINDEX_LINKAGE const char *clang_getDeclSource(CXDecl); /* deprecate */
CINDEX_LINKAGE CXFile clang_getDeclSourceFile(CXDecl); /* deprecate */

/**
 * \brief Identifies a specific source location within a translation
 * unit.
 *
 * Use clang_getInstantiationLocation() to map a source location to a
 * particular file, line, and column.
 */
typedef struct {
  void *ptr_data;
  unsigned int_data;
} CXSourceLocation;

/**
 * \brief Identifies a range of source locations in the source code.
 *
 * Use clang_getRangeStart() and clang_getRangeEnd() to retrieve the
 * starting and end locations from a source range, respectively.
 */
typedef struct {
  void *ptr_data;
  unsigned begin_int_data;
  unsigned end_int_data;
} CXSourceRange;

/**
 * \brief Retrieve the file, line, and column represented by the
 * given source location.
 *
 * \param location the location within a source file that will be
 * decomposed into its parts.
 *
 * \param file if non-NULL, will be set to the file to which the given
 * source location points.
 *
 * \param line if non-NULL, will be set to the line to which the given
 * source location points.
 *
 * \param column if non-NULL, will be set to the column to which the
 * given source location points.
 */
CINDEX_LINKAGE void clang_getInstantiationLocation(CXSourceLocation location,
                                                   CXFile *file,
                                                   unsigned *line,
                                                   unsigned *column);

/**
 * \brief Retrieve a source location representing the first
 * character within a source range.
 */
CINDEX_LINKAGE CXSourceLocation clang_getRangeStart(CXSourceRange range);

/**
 * \brief Retrieve a source location representing the last
 * character within a source range.
 */
CINDEX_LINKAGE CXSourceLocation clang_getRangeEnd(CXSourceRange range);

/* clang_getDeclExtent() returns the physical extent of a declaration.  The
 * beginning line/column pair points to the start of the first token in the
 * declaration, and the ending line/column pair points to the last character in
 * the last token of the declaration.
 */
CINDEX_LINKAGE CXSourceRange clang_getDeclExtent(CXDecl);

/*
 * CXCursor Operations.
 */
/**
   Usage: clang_getCursor() will translate a source/line/column position
   into an AST cursor (to derive semantic information from the source code).
 */
CINDEX_LINKAGE CXCursor clang_getCursor(CXTranslationUnit,
                                        const char *source_name, 
                                        unsigned line, unsigned column);
                         
CINDEX_LINKAGE CXCursor clang_getNullCursor(void);

/* clang_getCursorUSR() returns the USR (if any) associated with entity referred to by the
 *   provided CXCursor object. */
CINDEX_LINKAGE CXString clang_getCursorUSR(CXCursor);

CINDEX_LINKAGE enum CXCursorKind clang_getCursorKind(CXCursor);
CINDEX_LINKAGE unsigned clang_isDeclaration(enum CXCursorKind);
CINDEX_LINKAGE unsigned clang_isReference(enum CXCursorKind);
CINDEX_LINKAGE unsigned clang_isInvalid(enum CXCursorKind);

CINDEX_LINKAGE unsigned clang_equalCursors(CXCursor, CXCursor);

CINDEX_LINKAGE CXString clang_getCursorSpelling(CXCursor);

/**
 * \brief Retrieve the physical location of the source constructor referenced
 * by the given cursor.
 *
 * The location of a declaration is typically the location of the name of that
 * declaration, where the name of that declaration would occur if it is 
 * unnamed, or some keyword that introduces that particular declaration. 
 * The location of a reference is where that reference occurs within the 
 * source code.
 */
CINDEX_LINKAGE CXSourceLocation clang_getCursorLocation(CXCursor);
    
/**
 * \brief Retrieve the physical extent of the source construct referenced by
 * the given cursor.
 *
 * The extent of a cursor starts with the file/line/column pointing at the
 * first character within the source construct that the cursor refers to and
 * ends with the last character withinin that source construct. For a 
 * declaration, the extent covers the declaration itself. For a reference,
 * the extent covers the location of the reference (e.g., where the referenced
 * entity was actually used).
 */
CINDEX_LINKAGE CXSourceRange clang_getCursorExtent(CXCursor);

/** \brief For a cursor that is a reference, retrieve a cursor representing the
 * entity that it references.
 *
 * Reference cursors refer to other entities in the AST. For example, an
 * Objective-C superclass reference cursor refers to an Objective-C class.
 * This function produces the cursor for the Objective-C class from the 
 * cursor for the superclass reference. If the input cursor is a declaration or
 * definition, it returns that declaration or definition unchanged.
 * Othewise, returns the NULL cursor.
 */
CINDEX_LINKAGE CXCursor clang_getCursorReferenced(CXCursor);

/** 
 *  \brief For a cursor that is either a reference to or a declaration
 *  of some entity, retrieve a cursor that describes the definition of
 *  that entity.
 *
 *  Some entities can be declared multiple times within a translation
 *  unit, but only one of those declarations can also be a
 *  definition. For example, given:
 *
 *  \code
 *  int f(int, int);
 *  int g(int x, int y) { return f(x, y); }
 *  int f(int a, int b) { return a + b; }
 *  int f(int, int);
 *  \endcode
 *
 *  there are three declarations of the function "f", but only the
 *  second one is a definition. The clang_getCursorDefinition()
 *  function will take any cursor pointing to a declaration of "f"
 *  (the first or fourth lines of the example) or a cursor referenced
 *  that uses "f" (the call to "f' inside "g") and will return a
 *  declaration cursor pointing to the definition (the second "f"
 *  declaration).
 *
 *  If given a cursor for which there is no corresponding definition,
 *  e.g., because there is no definition of that entity within this
 *  translation unit, returns a NULL cursor.
 */
CINDEX_LINKAGE CXCursor clang_getCursorDefinition(CXCursor);

/** 
 * \brief Determine whether the declaration pointed to by this cursor
 * is also a definition of that entity.
 */
CINDEX_LINKAGE unsigned clang_isCursorDefinition(CXCursor);

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

/**
 * \brief A semantic string that describes a code-completion result.
 *
 * A semantic string that describes the formatting of a code-completion
 * result as a single "template" of text that should be inserted into the
 * source buffer when a particular code-completion result is selected.
 * Each semantic string is made up of some number of "chunks", each of which
 * contains some text along with a description of what that text means, e.g.,
 * the name of the entity being referenced, whether the text chunk is part of
 * the template, or whether it is a "placeholder" that the user should replace
 * with actual code,of a specific kind. See \c CXCompletionChunkKind for a
 * description of the different kinds of chunks. 
 */
typedef void *CXCompletionString;
  
/**
 * \brief A single result of code completion.
 */
typedef struct {
  /**
   * \brief The kind of entity that this completion refers to. 
   *
   * The cursor kind will be a macro, keyword, or a declaration (one of the 
   * *Decl cursor kinds), describing the entity that the completion is
   * referring to.
   *
   * \todo In the future, we would like to provide a full cursor, to allow
   * the client to extract additional information from declaration.
   */
  enum CXCursorKind CursorKind;
  
  /** 
   * \brief The code-completion string that describes how to insert this
   * code-completion result into the editing buffer.
   */
  CXCompletionString CompletionString;
} CXCompletionResult;

/**
 * \brief Describes a single piece of text within a code-completion string.
 *
 * Each "chunk" within a code-completion string (\c CXCompletionString) is 
 * either a piece of text with a specific "kind" that describes how that text 
 * should be interpreted by the client or is another completion string.
 */
enum CXCompletionChunkKind {
  /**
   * \brief A code-completion string that describes "optional" text that
   * could be a part of the template (but is not required).
   *
   * The Optional chunk is the only kind of chunk that has a code-completion
   * string for its representation, which is accessible via 
   * \c clang_getCompletionChunkCompletionString(). The code-completion string
   * describes an additional part of the template that is completely optional.
   * For example, optional chunks can be used to describe the placeholders for
   * arguments that match up with defaulted function parameters, e.g. given:
   *
   * \code
   * void f(int x, float y = 3.14, double z = 2.71828);
   * \endcode
   *
   * The code-completion string for this function would contain:
   *   - a TypedText chunk for "f".
   *   - a LeftParen chunk for "(".
   *   - a Placeholder chunk for "int x"
   *   - an Optional chunk containing the remaining defaulted arguments, e.g.,
   *       - a Comma chunk for ","
   *       - a Placeholder chunk for "float x"
   *       - an Optional chunk containing the last defaulted argument:
   *           - a Comma chunk for ","
   *           - a Placeholder chunk for "double z"
   *   - a RightParen chunk for ")"
   *
   * There are many ways two handle Optional chunks. Two simple approaches are:
   *   - Completely ignore optional chunks, in which case the template for the
   *     function "f" would only include the first parameter ("int x").
   *   - Fully expand all optional chunks, in which case the template for the
   *     function "f" would have all of the parameters.
   */
  CXCompletionChunk_Optional,
  /**
   * \brief Text that a user would be expected to type to get this
   * code-completion result. 
   *
   * There will be exactly one "typed text" chunk in a semantic string, which 
   * will typically provide the spelling of a keyword or the name of a 
   * declaration that could be used at the current code point. Clients are
   * expected to filter the code-completion results based on the text in this
   * chunk.
   */
  CXCompletionChunk_TypedText,
  /**
   * \brief Text that should be inserted as part of a code-completion result.
   *
   * A "text" chunk represents text that is part of the template to be
   * inserted into user code should this particular code-completion result
   * be selected.
   */
  CXCompletionChunk_Text,
  /**
   * \brief Placeholder text that should be replaced by the user.
   *
   * A "placeholder" chunk marks a place where the user should insert text
   * into the code-completion template. For example, placeholders might mark
   * the function parameters for a function declaration, to indicate that the
   * user should provide arguments for each of those parameters. The actual
   * text in a placeholder is a suggestion for the text to display before
   * the user replaces the placeholder with real code.
   */
  CXCompletionChunk_Placeholder,
  /**
   * \brief Informative text that should be displayed but never inserted as
   * part of the template.
   * 
   * An "informative" chunk contains annotations that can be displayed to
   * help the user decide whether a particular code-completion result is the
   * right option, but which is not part of the actual template to be inserted
   * by code completion.
   */
  CXCompletionChunk_Informative,
  /**
   * \brief Text that describes the current parameter when code-completion is
   * referring to function call, message send, or template specialization.
   *
   * A "current parameter" chunk occurs when code-completion is providing
   * information about a parameter corresponding to the argument at the
   * code-completion point. For example, given a function
   *
   * \code
   * int add(int x, int y);
   * \endcode
   *
   * and the source code \c add(, where the code-completion point is after the
   * "(", the code-completion string will contain a "current parameter" chunk
   * for "int x", indicating that the current argument will initialize that
   * parameter. After typing further, to \c add(17, (where the code-completion
   * point is after the ","), the code-completion string will contain a 
   * "current paremeter" chunk to "int y".
   */
  CXCompletionChunk_CurrentParameter,
  /**
   * \brief A left parenthesis ('('), used to initiate a function call or
   * signal the beginning of a function parameter list.
   */
  CXCompletionChunk_LeftParen,
  /**
   * \brief A right parenthesis (')'), used to finish a function call or
   * signal the end of a function parameter list.
   */
  CXCompletionChunk_RightParen,
  /**
   * \brief A left bracket ('[').
   */
  CXCompletionChunk_LeftBracket,
  /**
   * \brief A right bracket (']').
   */
  CXCompletionChunk_RightBracket,
  /**
   * \brief A left brace ('{').
   */
  CXCompletionChunk_LeftBrace,
  /**
   * \brief A right brace ('}').
   */
  CXCompletionChunk_RightBrace,
  /**
   * \brief A left angle bracket ('<').
   */
  CXCompletionChunk_LeftAngle,
  /**
   * \brief A right angle bracket ('>').
   */
  CXCompletionChunk_RightAngle,
  /**
   * \brief A comma separator (',').
   */
  CXCompletionChunk_Comma,
  /**
   * \brief Text that specifies the result type of a given result. 
   *
   * This special kind of informative chunk is not meant to be inserted into
   * the text buffer. Rather, it is meant to illustrate the type that an 
   * expression using the given completion string would have.
   */
  CXCompletionChunk_ResultType,
  /**
   * \brief A colon (':').
   */
  CXCompletionChunk_Colon,
  /**
   * \brief A semicolon (';').
   */
  CXCompletionChunk_SemiColon,
  /**
   * \brief An '=' sign.
   */
  CXCompletionChunk_Equal,
  /**
   * Horizontal space (' ').
   */
  CXCompletionChunk_HorizontalSpace,
  /**
   * Vertical space ('\n'), after which it is generally a good idea to
   * perform indentation.
   */
  CXCompletionChunk_VerticalSpace
};
  
/**
 * \brief Determine the kind of a particular chunk within a completion string.
 *
 * \param completion_string the completion string to query.
 *
 * \param chunk_number the 0-based index of the chunk in the completion string.
 *
 * \returns the kind of the chunk at the index \c chunk_number.
 */
CINDEX_LINKAGE enum CXCompletionChunkKind 
clang_getCompletionChunkKind(CXCompletionString completion_string,
                             unsigned chunk_number);
  
/**
 * \brief Retrieve the text associated with a particular chunk within a 
 * completion string.
 *
 * \param completion_string the completion string to query.
 *
 * \param chunk_number the 0-based index of the chunk in the completion string.
 *
 * \returns the text associated with the chunk at index \c chunk_number.
 */
CINDEX_LINKAGE const char *
clang_getCompletionChunkText(CXCompletionString completion_string,
                             unsigned chunk_number);

/**
 * \brief Retrieve the completion string associated with a particular chunk 
 * within a completion string.
 *
 * \param completion_string the completion string to query.
 *
 * \param chunk_number the 0-based index of the chunk in the completion string.
 *
 * \returns the completion string associated with the chunk at index
 * \c chunk_number, or NULL if that chunk is not represented by a completion
 * string.
 */
CINDEX_LINKAGE CXCompletionString
clang_getCompletionChunkCompletionString(CXCompletionString completion_string,
                                         unsigned chunk_number);
  
/**
 * \brief Retrieve the number of chunks in the given code-completion string.
 */
CINDEX_LINKAGE unsigned
clang_getNumCompletionChunks(CXCompletionString completion_string);

/**
 * \brief Contains the results of code-completion.
 *
 * This data structure contains the results of code completion, as
 * produced by \c clang_codeComplete. Its contents must be freed by 
 * \c clang_disposeCodeCompleteResults.
 */
typedef struct {
  /**
   * \brief The code-completion results.
   */
  CXCompletionResult *Results;

  /**
   * \brief The number of code-completion results stored in the
   * \c Results array.
   */
  unsigned NumResults;
} CXCodeCompleteResults;

/**
 * \brief Perform code completion at a given location in a source file.
 *
 * This function performs code completion at a particular file, line, and
 * column within source code, providing results that suggest potential
 * code snippets based on the context of the completion. The basic model
 * for code completion is that Clang will parse a complete source file,
 * performing syntax checking up to the location where code-completion has
 * been requested. At that point, a special code-completion token is passed
 * to the parser, which recognizes this token and determines, based on the
 * current location in the C/Objective-C/C++ grammar and the state of 
 * semantic analysis, what completions to provide. These completions are
 * returned via a new \c CXCodeCompleteResults structure.
 *
 * Code completion itself is meant to be triggered by the client when the
 * user types punctuation characters or whitespace, at which point the 
 * code-completion location will coincide with the cursor. For example, if \c p
 * is a pointer, code-completion might be triggered after the "-" and then
 * after the ">" in \c p->. When the code-completion location is afer the ">",
 * the completion results will provide, e.g., the members of the struct that
 * "p" points to. The client is responsible for placing the cursor at the
 * beginning of the token currently being typed, then filtering the results
 * based on the contents of the token. For example, when code-completing for
 * the expression \c p->get, the client should provide the location just after
 * the ">" (e.g., pointing at the "g") to this code-completion hook. Then, the
 * client can filter the results based on the current token text ("get"), only
 * showing those results that start with "get". The intent of this interface
 * is to separate the relatively high-latency acquisition of code-completion
 * results from the filtering of results on a per-character basis, which must
 * have a lower latency.
 *
 * \param CIdx the \c CXIndex instance that will be used to perform code
 * completion.
 *
 * \param source_filename the name of the source file that should be parsed to
 * perform code-completion. This source file must be the same as or include the
 * filename described by \p complete_filename, or no code-completion results
 * will be produced.  NOTE: One can also specify NULL for this argument if the
 * source file is included in command_line_args.
 *
 * \param num_command_line_args the number of command-line arguments stored in
 * \p command_line_args.
 *
 * \param command_line_args the command-line arguments to pass to the Clang
 * compiler to build the given source file. This should include all of the 
 * necessary include paths, language-dialect switches, precompiled header
 * includes, etc., but should not include any information specific to 
 * code completion.
 *
 * \param num_unsaved_files the number of unsaved file entries in \p
 * unsaved_files.
 *
 * \param unsaved_files the files that have not yet been saved to disk
 * but may be required for code completion, including the contents of
 * those files.
 *
 * \param complete_filename the name of the source file where code completion
 * should be performed. In many cases, this name will be the same as the
 * source filename. However, the completion filename may also be a file 
 * included by the source file, which is required when producing 
 * code-completion results for a header.
 *
 * \param complete_line the line at which code-completion should occur.
 *
 * \param complete_column the column at which code-completion should occur. 
 * Note that the column should point just after the syntactic construct that
 * initiated code completion, and not in the middle of a lexical token.
 *
 * \returns if successful, a new CXCodeCompleteResults structure
 * containing code-completion results, which should eventually be
 * freed with \c clang_disposeCodeCompleteResults(). If code
 * completion fails, returns NULL.
 */
CINDEX_LINKAGE 
CXCodeCompleteResults *clang_codeComplete(CXIndex CIdx, 
                                          const char *source_filename,
                                          int num_command_line_args, 
                                          const char **command_line_args,
                                          unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files,
                                          const char *complete_filename,
                                          unsigned complete_line,
                                          unsigned complete_column);
  
/**
 * \brief Free the given set of code-completion results.
 */
CINDEX_LINKAGE 
void clang_disposeCodeCompleteResults(CXCodeCompleteResults *Results);
  
#ifdef __cplusplus
}
#endif
#endif

