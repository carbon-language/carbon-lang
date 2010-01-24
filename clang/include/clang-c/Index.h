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

/** \defgroup CINDEX C Interface to Clang
 *
 * The C Interface to Clang provides a relatively small API that exposes
 * facilities for parsing source code into an abstract syntax tree (AST),
 * loading already-parsed ASTs, traversing the AST, associating
 * physical source locations with elements within the AST, and other
 * facilities that support Clang-based development tools.
 *
 * This C interface to Clang will never provide all of the information
 * representation stored in Clang's C++ AST, nor should it: the intent is to
 * maintain an API that is relatively stable from one release to the next,
 * providing only the basic functionality needed to support development tools.
 *
 * To avoid namespace pollution, data types are prefixed with "CX" and
 * functions are prefixed with "clang_".
 *
 * @{
 */

/**
 * \brief An "index" that consists of a set of translation units that would
 * typically be linked together into an executable or library.
 */
typedef void *CXIndex;

/**
 * \brief A single translation unit, which resides in an index.
 */
typedef void *CXTranslationUnit;  /* A translation unit instance. */

/**
 * \brief Opaque pointer representing client data that will be passed through
 * to various callbacks and visitors.
 */
typedef void *CXClientData;

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

/**
 * \defgroup CINDEX_STRING String manipulation routines
 *
 * @{
 */

/**
 * \brief A character string.
 *
 * The \c CXString type is used to return strings from the interface when
 * the ownership of that string might different from one call to the next.
 * Use \c clang_getCString() to retrieve the string data and, once finished
 * with the string data, call \c clang_disposeString() to free the string.
 */
typedef struct {
  const char *Spelling;
  /* A 1 value indicates the clang_ indexing API needed to allocate the string
     (and it must be freed by clang_disposeString()). */
  int MustFreeString;
} CXString;

/**
 * \brief Retrieve the character data associated with the given string.
 */
CINDEX_LINKAGE const char *clang_getCString(CXString string);

/**
 * \brief Free the given string,
 */
CINDEX_LINKAGE void clang_disposeString(CXString string);

/**
 * @}
 */

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
 *   clang_visitChildren(clang_getTranslationUnitCursor(TU),
 *                       TranslationUnitVisitor, 0);
 *   clang_disposeTranslationUnit(TU);
 *
 *   // This will load all the symbols from 'IndexTest.c', excluding symbols
 *   // from 'IndexTest.pch'.
 *   char *args[] = { "-Xclang", "-include-pch=IndexTest.pch", 0 };
 *   TU = clang_createTranslationUnitFromSourceFile(Idx, "IndexTest.c", 2, args);
 *   clang_visitChildren(clang_getTranslationUnitCursor(TU),
 *                       TranslationUnitVisitor, 0);
 *   clang_disposeTranslationUnit(TU);
 *
 * This process of creating the 'pch', loading it separately, and using it (via
 * -include-pch) allows 'excludeDeclsFromPCH' to remove redundant callbacks
 * (which gives the indexer the same performance benefit as the compiler).
 */
CINDEX_LINKAGE CXIndex clang_createIndex(int excludeDeclarationsFromPCH,
                          int displayDiagnostics);
CINDEX_LINKAGE void clang_disposeIndex(CXIndex index);

/**
 * \brief Get the original translation unit source file name.
 */
CINDEX_LINKAGE CXString
clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit);

/**
 * \brief Request that AST's be generated externally for API calls which parse
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

/**
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
 *
 * \param num_unsaved_files the number of unsaved file entries in \p
 * unsaved_files.
 *
 * \param unsaved_files the files that have not yet been saved to disk
 * but may be required for code completion, including the contents of
 * those files.
 */
CINDEX_LINKAGE CXTranslationUnit clang_createTranslationUnitFromSourceFile(
                                        CXIndex CIdx,
                                        const char *source_filename,
                                        int num_clang_command_line_args,
                                        const char **clang_command_line_args,
                                        unsigned num_unsaved_files,
                                        struct CXUnsavedFile *unsaved_files);

/**
 * \defgroup CINDEX_FILES File manipulation routines
 *
 * @{
 */

/**
 * \brief A particular source file that is part of a translation unit.
 */
typedef void *CXFile;


/**
 * \brief Retrieve the complete file and path name of the given file.
 */
CINDEX_LINKAGE const char *clang_getFileName(CXFile SFile);

/**
 * \brief Retrieve the last modification time of the given file.
 */
CINDEX_LINKAGE time_t clang_getFileTime(CXFile SFile);

/**
 * \brief Retrieve a file handle within the given translation unit.
 *
 * \param tu the translation unit
 *
 * \param file_name the name of the file.
 *
 * \returns the file handle for the named file in the translation unit \p tu,
 * or a NULL file handle if the file was not a part of this translation unit.
 */
CINDEX_LINKAGE CXFile clang_getFile(CXTranslationUnit tu,
                                    const char *file_name);

/**
 * @}
 */

/**
 * \defgroup CINDEX_LOCATIONS Physical source locations
 *
 * Clang represents physical source locations in its abstract syntax tree in
 * great detail, with file, line, and column information for the majority of
 * the tokens parsed in the source code. These data types and functions are
 * used to represent source location information, either for a particular
 * point in the program or for a range of points in the program, and extract
 * specific location information from those data types.
 *
 * @{
 */

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
 * \brief Retrieve a NULL (invalid) source location.
 */
CINDEX_LINKAGE CXSourceLocation clang_getNullLocation();

/**
 * \determine Determine whether two source locations, which must refer into
 * the same translation unit, refer to exactly the same point in the source
 * code.
 *
 * \returns non-zero if the source locations refer to the same location, zero
 * if they refer to different locations.
 */
CINDEX_LINKAGE unsigned clang_equalLocations(CXSourceLocation loc1,
                                             CXSourceLocation loc2);

/**
 * \brief Retrieves the source location associated with a given file/line/column
 * in a particular translation unit.
 */
CINDEX_LINKAGE CXSourceLocation clang_getLocation(CXTranslationUnit tu,
                                                  CXFile file,
                                                  unsigned line,
                                                  unsigned column);

/**
 * \brief Retrieve a source range given the beginning and ending source
 * locations.
 */
CINDEX_LINKAGE CXSourceRange clang_getRange(CXSourceLocation begin,
                                            CXSourceLocation end);

/**
 * \brief Retrieve the file, line, and column represented by the given source
 * location.
 *
 * \param location the location within a source file that will be decomposed
 * into its parts.
 *
 * \param file [out] if non-NULL, will be set to the file to which the given
 * source location points.
 *
 * \param line [out] if non-NULL, will be set to the line to which the given
 * source location points.
 *
 * \param column [out] if non-NULL, will be set to the column to which the given
 * source location points.
 */
CINDEX_LINKAGE void clang_getInstantiationLocation(CXSourceLocation location,
                                                   CXFile *file,
                                                   unsigned *line,
                                                   unsigned *column);

/**
 * \brief Retrieve a source location representing the first character within a
 * source range.
 */
CINDEX_LINKAGE CXSourceLocation clang_getRangeStart(CXSourceRange range);

/**
 * \brief Retrieve a source location representing the last character within a
 * source range.
 */
CINDEX_LINKAGE CXSourceLocation clang_getRangeEnd(CXSourceRange range);

/**
 * @}
 */

/**
 * \brief Describes the kind of entity that a cursor refers to.
 */
enum CXCursorKind {
  /* Declarations */
  CXCursor_FirstDecl                     = 1,
  /**
   * \brief A declaration whose specific kind is not exposed via this
   * interface.
   *
   * Unexposed declarations have the same operations as any other kind
   * of declaration; one can extract their location information,
   * spelling, find their definitions, etc. However, the specific kind
   * of the declaration is not reported.
   */
  CXCursor_UnexposedDecl                 = 1,
  /** \brief A C or C++ struct. */
  CXCursor_StructDecl                    = 2,
  /** \brief A C or C++ union. */
  CXCursor_UnionDecl                     = 3,
  /** \brief A C++ class. */
  CXCursor_ClassDecl                     = 4,
  /** \brief An enumeration. */
  CXCursor_EnumDecl                      = 5,
  /**
   * \brief A field (in C) or non-static data member (in C++) in a
   * struct, union, or C++ class.
   */
  CXCursor_FieldDecl                     = 6,
  /** \brief An enumerator constant. */
  CXCursor_EnumConstantDecl              = 7,
  /** \brief A function. */
  CXCursor_FunctionDecl                  = 8,
  /** \brief A variable. */
  CXCursor_VarDecl                       = 9,
  /** \brief A function or method parameter. */
  CXCursor_ParmDecl                      = 10,
  /** \brief An Objective-C @interface. */
  CXCursor_ObjCInterfaceDecl             = 11,
  /** \brief An Objective-C @interface for a category. */
  CXCursor_ObjCCategoryDecl              = 12,
  /** \brief An Objective-C @protocol declaration. */
  CXCursor_ObjCProtocolDecl              = 13,
  /** \brief An Objective-C @property declaration. */
  CXCursor_ObjCPropertyDecl              = 14,
  /** \brief An Objective-C instance variable. */
  CXCursor_ObjCIvarDecl                  = 15,
  /** \brief An Objective-C instance method. */
  CXCursor_ObjCInstanceMethodDecl        = 16,
  /** \brief An Objective-C class method. */
  CXCursor_ObjCClassMethodDecl           = 17,
  /** \brief An Objective-C @implementation. */
  CXCursor_ObjCImplementationDecl        = 18,
  /** \brief An Objective-C @implementation for a category. */
  CXCursor_ObjCCategoryImplDecl          = 19,
  /** \brief A typedef */
  CXCursor_TypedefDecl                   = 20,
  CXCursor_LastDecl                      = 20,

  /* References */
  CXCursor_FirstRef                      = 40, /* Decl references */
  CXCursor_ObjCSuperClassRef             = 40,
  CXCursor_ObjCProtocolRef               = 41,
  CXCursor_ObjCClassRef                  = 42,
  /**
   * \brief A reference to a type declaration.
   *
   * A type reference occurs anywhere where a type is named but not
   * declared. For example, given:
   *
   * \code
   * typedef unsigned size_type;
   * size_type size;
   * \endcode
   *
   * The typedef is a declaration of size_type (CXCursor_TypedefDecl),
   * while the type of the variable "size" is referenced. The cursor
   * referenced by the type of size is the typedef for size_type.
   */
  CXCursor_TypeRef                       = 43,
  CXCursor_LastRef                       = 43,

  /* Error conditions */
  CXCursor_FirstInvalid                  = 70,
  CXCursor_InvalidFile                   = 70,
  CXCursor_NoDeclFound                   = 71,
  CXCursor_NotImplemented                = 72,
  CXCursor_LastInvalid                   = 72,

  /* Expressions */
  CXCursor_FirstExpr                     = 100,

  /**
   * \brief An expression whose specific kind is not exposed via this
   * interface.
   *
   * Unexposed expressions have the same operations as any other kind
   * of expression; one can extract their location information,
   * spelling, children, etc. However, the specific kind of the
   * expression is not reported.
   */
  CXCursor_UnexposedExpr                 = 100,

  /**
   * \brief An expression that refers to some value declaration, such
   * as a function, varible, or enumerator.
   */
  CXCursor_DeclRefExpr                   = 101,

  /**
   * \brief An expression that refers to a member of a struct, union,
   * class, Objective-C class, etc.
   */
  CXCursor_MemberRefExpr                 = 102,

  /** \brief An expression that calls a function. */
  CXCursor_CallExpr                      = 103,

  /** \brief An expression that sends a message to an Objective-C
   object or class. */
  CXCursor_ObjCMessageExpr               = 104,
  CXCursor_LastExpr                      = 104,

  /* Statements */
  CXCursor_FirstStmt                     = 200,
  /**
   * \brief A statement whose specific kind is not exposed via this
   * interface.
   *
   * Unexposed statements have the same operations as any other kind of
   * statement; one can extract their location information, spelling,
   * children, etc. However, the specific kind of the statement is not
   * reported.
   */
  CXCursor_UnexposedStmt                 = 200,
  CXCursor_LastStmt                      = 200,

  /**
   * \brief Cursor that represents the translation unit itself.
   *
   * The translation unit cursor exists primarily to act as the root
   * cursor for traversing the contents of a translation unit.
   */
  CXCursor_TranslationUnit               = 300
};

/**
 * \brief A cursor representing some element in the abstract syntax tree for
 * a translation unit.
 *
 * The cursor abstraction unifies the different kinds of entities in a
 * program--declaration, statements, expressions, references to declarations,
 * etc.--under a single "cursor" abstraction with a common set of operations.
 * Common operation for a cursor include: getting the physical location in
 * a source file where the cursor points, getting the name associated with a
 * cursor, and retrieving cursors for any child nodes of a particular cursor.
 *
 * Cursors can be produced in two specific ways.
 * clang_getTranslationUnitCursor() produces a cursor for a translation unit,
 * from which one can use clang_visitChildren() to explore the rest of the
 * translation unit. clang_getCursor() maps from a physical source location
 * to the entity that resides at that location, allowing one to map from the
 * source code into the AST.
 */
typedef struct {
  enum CXCursorKind kind;
  void *data[3];
} CXCursor;

/**
 * \defgroup CINDEX_CURSOR_MANIP Cursor manipulations
 *
 * @{
 */

/**
 * \brief Retrieve the NULL cursor, which represents no entity.
 */
CINDEX_LINKAGE CXCursor clang_getNullCursor(void);

/**
 * \brief Retrieve the cursor that represents the given translation unit.
 *
 * The translation unit cursor can be used to start traversing the
 * various declarations within the given translation unit.
 */
CINDEX_LINKAGE CXCursor clang_getTranslationUnitCursor(CXTranslationUnit);

/**
 * \brief Determine whether two cursors are equivalent.
 */
CINDEX_LINKAGE unsigned clang_equalCursors(CXCursor, CXCursor);

/**
 * \brief Retrieve the kind of the given cursor.
 */
CINDEX_LINKAGE enum CXCursorKind clang_getCursorKind(CXCursor);

/**
 * \brief Determine whether the given cursor kind represents a declaration.
 */
CINDEX_LINKAGE unsigned clang_isDeclaration(enum CXCursorKind);

/**
 * \brief Determine whether the given cursor kind represents a simple
 * reference.
 *
 * Note that other kinds of cursors (such as expressions) can also refer to
 * other cursors. Use clang_getCursorReferenced() to determine whether a
 * particular cursor refers to another entity.
 */
CINDEX_LINKAGE unsigned clang_isReference(enum CXCursorKind);

/**
 * \brief Determine whether the given cursor kind represents an expression.
 */
CINDEX_LINKAGE unsigned clang_isExpression(enum CXCursorKind);

/**
 * \brief Determine whether the given cursor kind represents a statement.
 */
CINDEX_LINKAGE unsigned clang_isStatement(enum CXCursorKind);

/**
 * \brief Determine whether the given cursor kind represents an invalid
 * cursor.
 */
CINDEX_LINKAGE unsigned clang_isInvalid(enum CXCursorKind);

/**
 * \brief Determine whether the given cursor kind represents a translation
 * unit.
 */
CINDEX_LINKAGE unsigned clang_isTranslationUnit(enum CXCursorKind);

/**
 * @}
 */

/**
 * \defgroup CINDEX_CURSOR_SOURCE Mapping between cursors and source code
 *
 * Cursors represent a location within the Abstract Syntax Tree (AST). These
 * routines help map between cursors and the physical locations where the
 * described entities occur in the source code. The mapping is provided in
 * both directions, so one can map from source code to the AST and back.
 *
 * @{
 */

/**
 * \brief Map a source location to the cursor that describes the entity at that
 * location in the source code.
 *
 * clang_getCursor() maps an arbitrary source location within a translation
 * unit down to the most specific cursor that describes the entity at that
 * location. For example, given an expression \c x + y, invoking
 * clang_getCursor() with a source location pointing to "x" will return the
 * cursor for "x"; similarly for "y". If the cursor points anywhere between
 * "x" or "y" (e.g., on the + or the whitespace around it), clang_getCursor()
 * will return a cursor referring to the "+" expression.
 *
 * \returns a cursor representing the entity at the given source location, or
 * a NULL cursor if no such entity can be found.
 */
CINDEX_LINKAGE CXCursor clang_getCursor(CXTranslationUnit, CXSourceLocation);

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

/**
 * @}
 */

/**
 * \defgroup CINDEX_CURSOR_TRAVERSAL Traversing the AST with cursors
 *
 * These routines provide the ability to traverse the abstract syntax tree
 * using cursors.
 *
 * @{
 */

/**
 * \brief Describes how the traversal of the children of a particular
 * cursor should proceed after visiting a particular child cursor.
 *
 * A value of this enumeration type should be returned by each
 * \c CXCursorVisitor to indicate how clang_visitChildren() proceed.
 */
enum CXChildVisitResult {
  /**
   * \brief Terminates the cursor traversal.
   */
  CXChildVisit_Break,
  /**
   * \brief Continues the cursor traversal with the next sibling of
   * the cursor just visited, without visiting its children.
   */
  CXChildVisit_Continue,
  /**
   * \brief Recursively traverse the children of this cursor, using
   * the same visitor and client data.
   */
  CXChildVisit_Recurse
};

/**
 * \brief Visitor invoked for each cursor found by a traversal.
 *
 * This visitor function will be invoked for each cursor found by
 * clang_visitCursorChildren(). Its first argument is the cursor being
 * visited, its second argument is the parent visitor for that cursor,
 * and its third argument is the client data provided to
 * clang_visitCursorChildren().
 *
 * The visitor should return one of the \c CXChildVisitResult values
 * to direct clang_visitCursorChildren().
 */
typedef enum CXChildVisitResult (*CXCursorVisitor)(CXCursor cursor,
                                                   CXCursor parent,
                                                   CXClientData client_data);

/**
 * \brief Visit the children of a particular cursor.
 *
 * This function visits all the direct children of the given cursor,
 * invoking the given \p visitor function with the cursors of each
 * visited child. The traversal may be recursive, if the visitor returns
 * \c CXChildVisit_Recurse. The traversal may also be ended prematurely, if
 * the visitor returns \c CXChildVisit_Break.
 *
 * \param tu the translation unit into which the cursor refers.
 *
 * \param parent the cursor whose child may be visited. All kinds of
 * cursors can be visited, including invalid visitors (which, by
 * definition, have no children).
 *
 * \param visitor the visitor function that will be invoked for each
 * child of \p parent.
 *
 * \param client_data pointer data supplied by the client, which will
 * be passed to the visitor each time it is invoked.
 *
 * \returns a non-zero value if the traversal was terminated
 * prematurely by the visitor returning \c CXChildVisit_Break.
 */
CINDEX_LINKAGE unsigned clang_visitChildren(CXCursor parent,
                                            CXCursorVisitor visitor,
                                            CXClientData client_data);

/**
 * @}
 */

/**
 * \defgroup CINDEX_CURSOR_XREF Cross-referencing in the AST
 *
 * These routines provide the ability to determine references within and
 * across translation units, by providing the names of the entities referenced
 * by cursors, follow reference cursors to the declarations they reference,
 * and associate declarations with their definitions.
 *
 * @{
 */

/**
 * \brief Retrieve a Unified Symbol Resolution (USR) for the entity referenced
 * by the given cursor.
 *
 * A Unified Symbol Resolution (USR) is a string that identifies a particular
 * entity (function, class, variable, etc.) within a program. USRs can be
 * compared across translation units to determine, e.g., when references in
 * one translation refer to an entity defined in another translation unit.
 */
CINDEX_LINKAGE CXString clang_getCursorUSR(CXCursor);

/**
 * \brief Retrieve a name for the entity referenced by this cursor.
 */
CINDEX_LINKAGE CXString clang_getCursorSpelling(CXCursor);

/** \brief For a cursor that is a reference, retrieve a cursor representing the
 * entity that it references.
 *
 * Reference cursors refer to other entities in the AST. For example, an
 * Objective-C superclass reference cursor refers to an Objective-C class.
 * This function produces the cursor for the Objective-C class from the
 * cursor for the superclass reference. If the input cursor is a declaration or
 * definition, it returns that declaration or definition unchanged.
 * Otherwise, returns the NULL cursor.
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

/**
 * @}
 */

/**
 * \defgroup CINDEX_DEBUG Debugging facilities
 *
 * These routines are used for testing and debugging, only, and should not
 * be relied upon.
 *
 * @{
 */

/* for debug/testing */
CINDEX_LINKAGE const char *clang_getCursorKindSpelling(enum CXCursorKind Kind);
CINDEX_LINKAGE void clang_getDefinitionSpellingAndExtent(CXCursor,
                                          const char **startBuf,
                                          const char **endBuf,
                                          unsigned *startLine,
                                          unsigned *startColumn,
                                          unsigned *endLine,
                                          unsigned *endColumn);

/**
 * @}
 */

/**
 * \defgroup CINDEX_CODE_COMPLET Code completion
 *
 * Code completion involves taking an (incomplete) source file, along with
 * knowledge of where the user is actively editing that file, and suggesting
 * syntactically- and semantically-valid constructs that the user might want to
 * use at that particular point in the source code. These data structures and
 * routines provide support for code completion.
 *
 * @{
 */

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

/**
 * @}
 */


/**
 * \defgroup CINDEX_MISC Miscellaneous utility functions
 *
 * @{
 */

/**
 * \brief Return a version string, suitable for showing to a user, but not
 *        intended to be parsed (the format is not guaranteed to be stable).
 */
CINDEX_LINKAGE const char *clang_getClangVersion();

/**
 * @}
 */

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
#endif

