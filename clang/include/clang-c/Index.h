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
#include <stdio.h>

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
   * \brief A buffer containing the unsaved contents of this file.
   */
  const char *Contents;

  /**
   * \brief The length of the unsaved contents of this buffer.
   */
  unsigned long Length;
};

/**
 * \brief Describes the availability of a particular entity, which indicates
 * whether the use of this entity will result in a warning or error due to
 * it being deprecated or unavailable.
 */
enum CXAvailabilityKind {
  /**
   * \brief The entity is available.
   */
  CXAvailability_Available,
  /**
   * \brief The entity is available, but has been deprecated (and its use is
   * not recommended).
   */
  CXAvailability_Deprecated,
  /**
   * \brief The entity is not available; any use of it will be an error.
   */
  CXAvailability_NotAvailable
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
 * Here is an example:
 *
 *   // excludeDeclsFromPCH = 1, displayDiagnostics=1
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
 *   char *args[] = { "-Xclang", "-include-pch=IndexTest.pch" };
 *   TU = clang_createTranslationUnitFromSourceFile(Idx, "IndexTest.c", 2, args,
 *                                                  0, 0);
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

/**
 * \brief Destroy the given index.
 *
 * The index must not be destroyed until all of the translation units created
 * within that index have been destroyed.
 */
CINDEX_LINKAGE void clang_disposeIndex(CXIndex index);

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
CINDEX_LINKAGE CXString clang_getFileName(CXFile SFile);

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
  void *ptr_data[2];
  unsigned int_data;
} CXSourceLocation;

/**
 * \brief Identifies a half-open character range in the source code.
 *
 * Use clang_getRangeStart() and clang_getRangeEnd() to retrieve the
 * starting and end locations from a source range, respectively.
 */
typedef struct {
  void *ptr_data[2];
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
 * \brief Retrieve a NULL (invalid) source range.
 */
CINDEX_LINKAGE CXSourceRange clang_getNullRange();

/**
 * \brief Retrieve a source range given the beginning and ending source
 * locations.
 */
CINDEX_LINKAGE CXSourceRange clang_getRange(CXSourceLocation begin,
                                            CXSourceLocation end);

/**
 * \brief Retrieve the file, line, column, and offset represented by
 * the given source location.
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
 *
 * \param offset [out] if non-NULL, will be set to the offset into the
 * buffer to which the given source location points.
 */
CINDEX_LINKAGE void clang_getInstantiationLocation(CXSourceLocation location,
                                                   CXFile *file,
                                                   unsigned *line,
                                                   unsigned *column,
                                                   unsigned *offset);

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
 * \defgroup CINDEX_DIAG Diagnostic reporting
 *
 * @{
 */

/**
 * \brief Describes the severity of a particular diagnostic.
 */
enum CXDiagnosticSeverity {
  /**
   * \brief A diagnostic that has been suppressed, e.g., by a command-line
   * option.
   */
  CXDiagnostic_Ignored = 0,

  /**
   * \brief This diagnostic is a note that should be attached to the
   * previous (non-note) diagnostic.
   */
  CXDiagnostic_Note    = 1,

  /**
   * \brief This diagnostic indicates suspicious code that may not be
   * wrong.
   */
  CXDiagnostic_Warning = 2,

  /**
   * \brief This diagnostic indicates that the code is ill-formed.
   */
  CXDiagnostic_Error   = 3,

  /**
   * \brief This diagnostic indicates that the code is ill-formed such
   * that future parser recovery is unlikely to produce useful
   * results.
   */
  CXDiagnostic_Fatal   = 4
};

/**
 * \brief A single diagnostic, containing the diagnostic's severity,
 * location, text, source ranges, and fix-it hints.
 */
typedef void *CXDiagnostic;

/**
 * \brief Determine the number of diagnostics produced for the given
 * translation unit.
 */
CINDEX_LINKAGE unsigned clang_getNumDiagnostics(CXTranslationUnit Unit);

/**
 * \brief Retrieve a diagnostic associated with the given translation unit.
 *
 * \param Unit the translation unit to query.
 * \param Index the zero-based diagnostic number to retrieve.
 *
 * \returns the requested diagnostic. This diagnostic must be freed
 * via a call to \c clang_disposeDiagnostic().
 */
CINDEX_LINKAGE CXDiagnostic clang_getDiagnostic(CXTranslationUnit Unit,
                                                unsigned Index);

/**
 * \brief Destroy a diagnostic.
 */
CINDEX_LINKAGE void clang_disposeDiagnostic(CXDiagnostic Diagnostic);

/**
 * \brief Options to control the display of diagnostics.
 *
 * The values in this enum are meant to be combined to customize the
 * behavior of \c clang_displayDiagnostic().
 */
enum CXDiagnosticDisplayOptions {
  /**
   * \brief Display the source-location information where the
   * diagnostic was located.
   *
   * When set, diagnostics will be prefixed by the file, line, and
   * (optionally) column to which the diagnostic refers. For example,
   *
   * \code
   * test.c:28: warning: extra tokens at end of #endif directive
   * \endcode
   *
   * This option corresponds to the clang flag \c -fshow-source-location.
   */
  CXDiagnostic_DisplaySourceLocation = 0x01,

  /**
   * \brief If displaying the source-location information of the
   * diagnostic, also include the column number.
   *
   * This option corresponds to the clang flag \c -fshow-column.
   */
  CXDiagnostic_DisplayColumn = 0x02,

  /**
   * \brief If displaying the source-location information of the
   * diagnostic, also include information about source ranges in a
   * machine-parsable format.
   *
   * This option corresponds to the clang flag
   * \c -fdiagnostics-print-source-range-info.
   */
  CXDiagnostic_DisplaySourceRanges = 0x04
};

/**
 * \brief Format the given diagnostic in a manner that is suitable for display.
 *
 * This routine will format the given diagnostic to a string, rendering
 * the diagnostic according to the various options given. The
 * \c clang_defaultDiagnosticDisplayOptions() function returns the set of
 * options that most closely mimics the behavior of the clang compiler.
 *
 * \param Diagnostic The diagnostic to print.
 *
 * \param Options A set of options that control the diagnostic display,
 * created by combining \c CXDiagnosticDisplayOptions values.
 *
 * \returns A new string containing for formatted diagnostic.
 */
CINDEX_LINKAGE CXString clang_formatDiagnostic(CXDiagnostic Diagnostic,
                                               unsigned Options);

/**
 * \brief Retrieve the set of display options most similar to the
 * default behavior of the clang compiler.
 *
 * \returns A set of display options suitable for use with \c
 * clang_displayDiagnostic().
 */
CINDEX_LINKAGE unsigned clang_defaultDiagnosticDisplayOptions(void);

/**
 * \brief Print a diagnostic to the given file.
 */

/**
 * \brief Determine the severity of the given diagnostic.
 */
CINDEX_LINKAGE enum CXDiagnosticSeverity
clang_getDiagnosticSeverity(CXDiagnostic);

/**
 * \brief Retrieve the source location of the given diagnostic.
 *
 * This location is where Clang would print the caret ('^') when
 * displaying the diagnostic on the command line.
 */
CINDEX_LINKAGE CXSourceLocation clang_getDiagnosticLocation(CXDiagnostic);

/**
 * \brief Retrieve the text of the given diagnostic.
 */
CINDEX_LINKAGE CXString clang_getDiagnosticSpelling(CXDiagnostic);

/**
 * \brief Determine the number of source ranges associated with the given
 * diagnostic.
 */
CINDEX_LINKAGE unsigned clang_getDiagnosticNumRanges(CXDiagnostic);

/**
 * \brief Retrieve a source range associated with the diagnostic.
 *
 * A diagnostic's source ranges highlight important elements in the source
 * code. On the command line, Clang displays source ranges by
 * underlining them with '~' characters.
 *
 * \param Diagnostic the diagnostic whose range is being extracted.
 *
 * \param Range the zero-based index specifying which range to
 *
 * \returns the requested source range.
 */
CINDEX_LINKAGE CXSourceRange clang_getDiagnosticRange(CXDiagnostic Diagnostic,
                                                      unsigned Range);

/**
 * \brief Determine the number of fix-it hints associated with the
 * given diagnostic.
 */
CINDEX_LINKAGE unsigned clang_getDiagnosticNumFixIts(CXDiagnostic Diagnostic);

/**
 * \brief Retrieve the replacement information for a given fix-it.
 *
 * Fix-its are described in terms of a source range whose contents
 * should be replaced by a string. This approach generalizes over
 * three kinds of operations: removal of source code (the range covers
 * the code to be removed and the replacement string is empty),
 * replacement of source code (the range covers the code to be
 * replaced and the replacement string provides the new code), and
 * insertion (both the start and end of the range point at the
 * insertion location, and the replacement string provides the text to
 * insert).
 *
 * \param Diagnostic The diagnostic whose fix-its are being queried.
 *
 * \param FixIt The zero-based index of the fix-it.
 *
 * \param ReplacementRange The source range whose contents will be
 * replaced with the returned replacement string. Note that source
 * ranges are half-open ranges [a, b), so the source code should be
 * replaced from a and up to (but not including) b.
 *
 * \returns A string containing text that should be replace the source
 * code indicated by the \c ReplacementRange.
 */
CINDEX_LINKAGE CXString clang_getDiagnosticFixIt(CXDiagnostic Diagnostic,
                                                 unsigned FixIt,
                                               CXSourceRange *ReplacementRange);

/**
 * @}
 */

/**
 * \defgroup CINDEX_TRANSLATION_UNIT Translation unit manipulation
 *
 * The routines in this group provide the ability to create and destroy
 * translation units from files, either by parsing the contents of the files or
 * by reading in a serialized representation of a translation unit.
 *
 * @{
 */

/**
 * \brief Get the original translation unit source file name.
 */
CINDEX_LINKAGE CXString
clang_getTranslationUnitSpelling(CXTranslationUnit CTUnit);

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
 * those files.  The contents and name of these files (as specified by
 * CXUnsavedFile) are copied when necessary, so the client only needs to
 * guarantee their validity until the call to this function returns.
 *
 * \param diag_callback callback function that will receive any diagnostics
 * emitted while processing this source file. If NULL, diagnostics will be
 * suppressed.
 *
 * \param diag_client_data client data that will be passed to the diagnostic
 * callback function.
 */
CINDEX_LINKAGE CXTranslationUnit clang_createTranslationUnitFromSourceFile(
                                         CXIndex CIdx,
                                         const char *source_filename,
                                         int num_clang_command_line_args,
                                   const char * const *clang_command_line_args,
                                         unsigned num_unsaved_files,
                                         struct CXUnsavedFile *unsaved_files);

/**
 * \brief Create a translation unit from an AST file (-emit-ast).
 */
CINDEX_LINKAGE CXTranslationUnit clang_createTranslationUnit(CXIndex,
                                             const char *ast_filename);

/**
 * \brief Flags that control the creation of translation units.
 *
 * The enumerators in this enumeration type are meant to be bitwise
 * ORed together to specify which options should be used when
 * constructing the translation unit.
 */
enum CXTranslationUnit_Flags {
  /**
   * \brief Used to indicate that no special translation-unit options are
   * needed.
   */
  CXTranslationUnit_None = 0x0,

  /**
   * \brief Used to indicate that the parser should construct a "detailed"
   * preprocessing record, including all macro definitions and instantiations.
   *
   * Constructing a detailed preprocessing record requires more memory
   * and time to parse, since the information contained in the record
   * is usually not retained. However, it can be useful for
   * applications that require more detailed information about the
   * behavior of the preprocessor.
   */
  CXTranslationUnit_DetailedPreprocessingRecord = 0x01,

  /**
   * \brief Used to indicate that the translation unit is incomplete.
   *
   * When a translation unit is considered "incomplete", semantic
   * analysis that is typically performed at the end of the
   * translation unit will be suppressed. For example, this suppresses
   * the completion of tentative declarations in C and of
   * instantiation of implicitly-instantiation function templates in
   * C++. This option is typically used when parsing a header with the
   * intent of producing a precompiled header.
   */
  CXTranslationUnit_Incomplete = 0x02,
  
  /**
   * \brief Used to indicate that the translation unit should be built with an 
   * implicit precompiled header for the preamble.
   *
   * An implicit precompiled header is used as an optimization when a
   * particular translation unit is likely to be reparsed many times
   * when the sources aren't changing that often. In this case, an
   * implicit precompiled header will be built containing all of the
   * initial includes at the top of the main file (what we refer to as
   * the "preamble" of the file). In subsequent parses, if the
   * preamble or the files in it have not changed, \c
   * clang_reparseTranslationUnit() will re-use the implicit
   * precompiled header to improve parsing performance.
   */
  CXTranslationUnit_PrecompiledPreamble = 0x04,
  
  /**
   * \brief Used to indicate that the translation unit should cache some
   * code-completion results with each reparse of the source file.
   *
   * Caching of code-completion results is a performance optimization that
   * introduces some overhead to reparsing but improves the performance of
   * code-completion operations.
   */
  CXTranslationUnit_CacheCompletionResults = 0x08
};

/**
 * \brief Returns the set of flags that is suitable for parsing a translation
 * unit that is being edited.
 *
 * The set of flags returned provide options for \c clang_parseTranslationUnit()
 * to indicate that the translation unit is likely to be reparsed many times,
 * either explicitly (via \c clang_reparseTranslationUnit()) or implicitly
 * (e.g., by code completion (\c clang_codeCompletionAt())). The returned flag
 * set contains an unspecified set of optimizations (e.g., the precompiled 
 * preamble) geared toward improving the performance of these routines. The
 * set of optimizations enabled may change from one version to the next.
 */
CINDEX_LINKAGE unsigned clang_defaultEditingTranslationUnitOptions(void);
  
/**
 * \brief Parse the given source file and the translation unit corresponding
 * to that file.
 *
 * This routine is the main entry point for the Clang C API, providing the
 * ability to parse a source file into a translation unit that can then be
 * queried by other functions in the API. This routine accepts a set of
 * command-line arguments so that the compilation can be configured in the same
 * way that the compiler is configured on the command line.
 *
 * \param CIdx The index object with which the translation unit will be 
 * associated.
 *
 * \param source_filename The name of the source file to load, or NULL if the
 * source file is included in \p clang_command_line_args.
 *
 * \param command_line_args The command-line arguments that would be
 * passed to the \c clang executable if it were being invoked out-of-process.
 * These command-line options will be parsed and will affect how the translation
 * unit is parsed. Note that the following options are ignored: '-c', 
 * '-emit-ast', '-fsyntex-only' (which is the default), and '-o <output file>'.
 *
 * \param num_command_line_args The number of command-line arguments in
 * \p command_line_args.
 *
 * \param unsaved_files the files that have not yet been saved to disk
 * but may be required for parsing, including the contents of
 * those files.  The contents and name of these files (as specified by
 * CXUnsavedFile) are copied when necessary, so the client only needs to
 * guarantee their validity until the call to this function returns.
 *
 * \param num_unsaved_files the number of unsaved file entries in \p
 * unsaved_files.
 *
 * \param options A bitmask of options that affects how the translation unit
 * is managed but not its compilation. This should be a bitwise OR of the
 * CXTranslationUnit_XXX flags.
 *
 * \returns A new translation unit describing the parsed code and containing
 * any diagnostics produced by the compiler. If there is a failure from which
 * the compiler cannot recover, returns NULL.
 */
CINDEX_LINKAGE CXTranslationUnit clang_parseTranslationUnit(CXIndex CIdx,
                                                    const char *source_filename,
                                         const char * const *command_line_args,
                                                      int num_command_line_args,
                                            struct CXUnsavedFile *unsaved_files,
                                                     unsigned num_unsaved_files,
                                                            unsigned options);
  
/**
 * \brief Flags that control how translation units are saved.
 *
 * The enumerators in this enumeration type are meant to be bitwise
 * ORed together to specify which options should be used when
 * saving the translation unit.
 */
enum CXSaveTranslationUnit_Flags {
  /**
   * \brief Used to indicate that no special saving options are needed.
   */
  CXSaveTranslationUnit_None = 0x0
};

/**
 * \brief Returns the set of flags that is suitable for saving a translation
 * unit.
 *
 * The set of flags returned provide options for
 * \c clang_saveTranslationUnit() by default. The returned flag
 * set contains an unspecified set of options that save translation units with
 * the most commonly-requested data.
 */
CINDEX_LINKAGE unsigned clang_defaultSaveOptions(CXTranslationUnit TU);

/**
 * \brief Saves a translation unit into a serialized representation of
 * that translation unit on disk.
 *
 * Any translation unit that was parsed without error can be saved
 * into a file. The translation unit can then be deserialized into a
 * new \c CXTranslationUnit with \c clang_createTranslationUnit() or,
 * if it is an incomplete translation unit that corresponds to a
 * header, used as a precompiled header when parsing other translation
 * units.
 *
 * \param TU The translation unit to save.
 *
 * \param FileName The file to which the translation unit will be saved.
 *
 * \param options A bitmask of options that affects how the translation unit
 * is saved. This should be a bitwise OR of the
 * CXSaveTranslationUnit_XXX flags.
 *
 * \returns Zero if the translation unit was saved successfully, a
 * non-zero value otherwise.
 */
CINDEX_LINKAGE int clang_saveTranslationUnit(CXTranslationUnit TU,
                                             const char *FileName,
                                             unsigned options);

/**
 * \brief Destroy the specified CXTranslationUnit object.
 */
CINDEX_LINKAGE void clang_disposeTranslationUnit(CXTranslationUnit);

/**
 * \brief Flags that control the reparsing of translation units.
 *
 * The enumerators in this enumeration type are meant to be bitwise
 * ORed together to specify which options should be used when
 * reparsing the translation unit.
 */
enum CXReparse_Flags {
  /**
   * \brief Used to indicate that no special reparsing options are needed.
   */
  CXReparse_None = 0x0
};
 
/**
 * \brief Returns the set of flags that is suitable for reparsing a translation
 * unit.
 *
 * The set of flags returned provide options for
 * \c clang_reparseTranslationUnit() by default. The returned flag
 * set contains an unspecified set of optimizations geared toward common uses
 * of reparsing. The set of optimizations enabled may change from one version 
 * to the next.
 */
CINDEX_LINKAGE unsigned clang_defaultReparseOptions(CXTranslationUnit TU);

/**
 * \brief Reparse the source files that produced this translation unit.
 *
 * This routine can be used to re-parse the source files that originally
 * created the given translation unit, for example because those source files
 * have changed (either on disk or as passed via \p unsaved_files). The
 * source code will be reparsed with the same command-line options as it
 * was originally parsed. 
 *
 * Reparsing a translation unit invalidates all cursors and source locations
 * that refer into that translation unit. This makes reparsing a translation
 * unit semantically equivalent to destroying the translation unit and then
 * creating a new translation unit with the same command-line arguments.
 * However, it may be more efficient to reparse a translation 
 * unit using this routine.
 *
 * \param TU The translation unit whose contents will be re-parsed. The
 * translation unit must originally have been built with 
 * \c clang_createTranslationUnitFromSourceFile().
 *
 * \param num_unsaved_files The number of unsaved file entries in \p
 * unsaved_files.
 *
 * \param unsaved_files The files that have not yet been saved to disk
 * but may be required for parsing, including the contents of
 * those files.  The contents and name of these files (as specified by
 * CXUnsavedFile) are copied when necessary, so the client only needs to
 * guarantee their validity until the call to this function returns.
 * 
 * \param options A bitset of options composed of the flags in CXReparse_Flags.
 * The function \c clang_defaultReparseOptions() produces a default set of
 * options recommended for most uses, based on the translation unit.
 *
 * \returns 0 if the sources could be reparsed. A non-zero value will be
 * returned if reparsing was impossible, such that the translation unit is
 * invalid. In such cases, the only valid call for \p TU is 
 * \c clang_disposeTranslationUnit(TU).
 */
CINDEX_LINKAGE int clang_reparseTranslationUnit(CXTranslationUnit TU,
                                                unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files,
                                                unsigned options);
  
/**
 * @}
 */

/**
 * \brief Describes the kind of entity that a cursor refers to.
 */
enum CXCursorKind {
  /* Declarations */
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
  /** \brief A C++ class method. */
  CXCursor_CXXMethod                     = 21,
  /** \brief A C++ namespace. */
  CXCursor_Namespace                     = 22,
  /** \brief A linkage specification, e.g. 'extern "C"'. */
  CXCursor_LinkageSpec                   = 23,
  /** \brief A C++ constructor. */
  CXCursor_Constructor                   = 24,
  /** \brief A C++ destructor. */
  CXCursor_Destructor                    = 25,
  /** \brief A C++ conversion function. */
  CXCursor_ConversionFunction            = 26,
  /** \brief A C++ template type parameter. */
  CXCursor_TemplateTypeParameter         = 27,
  /** \brief A C++ non-type template parameter. */
  CXCursor_NonTypeTemplateParameter      = 28,
  /** \brief A C++ template template parameter. */
  CXCursor_TemplateTemplateParameter     = 29,
  /** \brief A C++ function template. */
  CXCursor_FunctionTemplate              = 30,
  /** \brief A C++ class template. */
  CXCursor_ClassTemplate                 = 31,
  /** \brief A C++ class template partial specialization. */
  CXCursor_ClassTemplatePartialSpecialization = 32,
  /** \brief A C++ namespace alias declaration. */
  CXCursor_NamespaceAlias                = 33,
  /** \brief A C++ using directive. */
  CXCursor_UsingDirective                = 34,
  
  CXCursor_FirstDecl                     = CXCursor_UnexposedDecl,
  CXCursor_LastDecl                      = CXCursor_UsingDirective,

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
  CXCursor_CXXBaseSpecifier              = 44,
  /** 
   * \brief A reference to a class template, function template, or template
   * template parameter. 
   */
  CXCursor_TemplateRef                   = 45,
  /**
   * \brief A reference to a namespace or namespace alias.
   */
  CXCursor_NamespaceRef                  = 46,
  CXCursor_LastRef                       = CXCursor_NamespaceRef,

  /* Error conditions */
  CXCursor_FirstInvalid                  = 70,
  CXCursor_InvalidFile                   = 70,
  CXCursor_NoDeclFound                   = 71,
  CXCursor_NotImplemented                = 72,
  CXCursor_InvalidCode                   = 73,
  CXCursor_LastInvalid                   = CXCursor_InvalidCode,

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

  /** \brief An expression that represents a block literal. */
  CXCursor_BlockExpr                     = 105,

  CXCursor_LastExpr                      = 105,

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
  CXCursor_TranslationUnit               = 300,

  /* Attributes */
  CXCursor_FirstAttr                     = 400,
  /**
   * \brief An attribute whose specific kind is not exposed via this
   * interface.
   */
  CXCursor_UnexposedAttr                 = 400,

  CXCursor_IBActionAttr                  = 401,
  CXCursor_IBOutletAttr                  = 402,
  CXCursor_IBOutletCollectionAttr        = 403,
  CXCursor_LastAttr                      = CXCursor_IBOutletCollectionAttr,
     
  /* Preprocessing */
  CXCursor_PreprocessingDirective        = 500,
  CXCursor_MacroDefinition               = 501,
  CXCursor_MacroInstantiation            = 502,
  CXCursor_FirstPreprocessing            = CXCursor_PreprocessingDirective,
  CXCursor_LastPreprocessing             = CXCursor_MacroInstantiation
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

/***
 * \brief Determine whether the given cursor represents a preprocessing
 * element, such as a preprocessor directive or macro instantiation.
 */
CINDEX_LINKAGE unsigned clang_isPreprocessing(enum CXCursorKind);
  
/***
 * \brief Determine whether the given cursor represents a currently
 *  unexposed piece of the AST (e.g., CXCursor_UnexposedStmt).
 */
CINDEX_LINKAGE unsigned clang_isUnexposed(enum CXCursorKind);

/**
 * \brief Describe the linkage of the entity referred to by a cursor.
 */
enum CXLinkageKind {
  /** \brief This value indicates that no linkage information is available
   * for a provided CXCursor. */
  CXLinkage_Invalid,
  /**
   * \brief This is the linkage for variables, parameters, and so on that
   *  have automatic storage.  This covers normal (non-extern) local variables.
   */
  CXLinkage_NoLinkage,
  /** \brief This is the linkage for static variables and static functions. */
  CXLinkage_Internal,
  /** \brief This is the linkage for entities with external linkage that live
   * in C++ anonymous namespaces.*/
  CXLinkage_UniqueExternal,
  /** \brief This is the linkage for entities with true, external linkage. */
  CXLinkage_External
};

/**
 * \brief Determine the linkage of the entity referred to by a given cursor.
 */
CINDEX_LINKAGE enum CXLinkageKind clang_getCursorLinkage(CXCursor cursor);

/**
 * \brief Determine the availability of the entity that this cursor refers to.
 *
 * \param cursor The cursor to query.
 *
 * \returns The availability of the cursor.
 */
CINDEX_LINKAGE enum CXAvailabilityKind 
clang_getCursorAvailability(CXCursor cursor);

/**
 * \brief Describe the "language" of the entity referred to by a cursor.
 */
CINDEX_LINKAGE enum CXLanguageKind {
  CXLanguage_Invalid = 0,
  CXLanguage_C,
  CXLanguage_ObjC,
  CXLanguage_CPlusPlus
};

/**
 * \brief Determine the "language" of the entity referred to by a given cursor.
 */
CINDEX_LINKAGE enum CXLanguageKind clang_getCursorLanguage(CXCursor cursor);

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
 * \defgroup CINDEX_TYPES Type information for CXCursors
 *
 * @{
 */

/**
 * \brief Describes the kind of type
 */
enum CXTypeKind {
  /**
   * \brief Reprents an invalid type (e.g., where no type is available).
   */
  CXType_Invalid = 0,

  /**
   * \brief A type whose specific kind is not exposed via this
   * interface.
   */
  CXType_Unexposed = 1,

  /* Builtin types */
  CXType_Void = 2,
  CXType_Bool = 3,
  CXType_Char_U = 4,
  CXType_UChar = 5,
  CXType_Char16 = 6,
  CXType_Char32 = 7,
  CXType_UShort = 8,
  CXType_UInt = 9,
  CXType_ULong = 10,
  CXType_ULongLong = 11,
  CXType_UInt128 = 12,
  CXType_Char_S = 13,
  CXType_SChar = 14,
  CXType_WChar = 15,
  CXType_Short = 16,
  CXType_Int = 17,
  CXType_Long = 18,
  CXType_LongLong = 19,
  CXType_Int128 = 20,
  CXType_Float = 21,
  CXType_Double = 22,
  CXType_LongDouble = 23,
  CXType_NullPtr = 24,
  CXType_Overload = 25,
  CXType_Dependent = 26,
  CXType_ObjCId = 27,
  CXType_ObjCClass = 28,
  CXType_ObjCSel = 29,
  CXType_FirstBuiltin = CXType_Void,
  CXType_LastBuiltin  = CXType_ObjCSel,

  CXType_Complex = 100,
  CXType_Pointer = 101,
  CXType_BlockPointer = 102,
  CXType_LValueReference = 103,
  CXType_RValueReference = 104,
  CXType_Record = 105,
  CXType_Enum = 106,
  CXType_Typedef = 107,
  CXType_ObjCInterface = 108,
  CXType_ObjCObjectPointer = 109,
  CXType_FunctionNoProto = 110,
  CXType_FunctionProto = 111
};

/**
 * \brief The type of an element in the abstract syntax tree.
 *
 */
typedef struct {
  enum CXTypeKind kind;
  void *data[2];
} CXType;

/**
 * \brief Retrieve the type of a CXCursor (if any).
 */
CINDEX_LINKAGE CXType clang_getCursorType(CXCursor C);

/**
 * \determine Determine whether two CXTypes represent the same type.
 *
 * \returns non-zero if the CXTypes represent the same type and 
            zero otherwise.
 */
CINDEX_LINKAGE unsigned clang_equalTypes(CXType A, CXType B);

/**
 * \brief Return the canonical type for a CXType.
 *
 * Clang's type system explicitly models typedefs and all the ways
 * a specific type can be represented.  The canonical type is the underlying
 * type with all the "sugar" removed.  For example, if 'T' is a typedef
 * for 'int', the canonical type for 'T' would be 'int'.
 */
CINDEX_LINKAGE CXType clang_getCanonicalType(CXType T);

/**
 * \brief For pointer types, returns the type of the pointee.
 *
 */
CINDEX_LINKAGE CXType clang_getPointeeType(CXType T);

/**
 * \brief Return the cursor for the declaration of the given type.
 */
CINDEX_LINKAGE CXCursor clang_getTypeDeclaration(CXType T);


/**
 * \brief Retrieve the spelling of a given CXTypeKind.
 */
CINDEX_LINKAGE CXString clang_getTypeKindSpelling(enum CXTypeKind K);

/**
 * \brief Retrieve the result type associated with a function type.
 */
CINDEX_LINKAGE CXType clang_getResultType(CXType T);

/**
 * \brief Retrieve the result type associated with a given cursor.  This only
 *  returns a valid type of the cursor refers to a function or method.
 */
CINDEX_LINKAGE CXType clang_getCursorResultType(CXCursor C);

/**
 * \brief Return 1 if the CXType is a POD (plain old data) type, and 0
 *  otherwise.
 */
CINDEX_LINKAGE unsigned clang_isPODType(CXType T);

/**
 * \brief Returns 1 if the base class specified by the cursor with kind
 *   CX_CXXBaseSpecifier is virtual.
 */
CINDEX_LINKAGE unsigned clang_isVirtualBase(CXCursor);
    
/**
 * \brief Represents the C++ access control level to a base class for a
 * cursor with kind CX_CXXBaseSpecifier.
 */
enum CX_CXXAccessSpecifier {
  CX_CXXInvalidAccessSpecifier,
  CX_CXXPublic,
  CX_CXXProtected,
  CX_CXXPrivate
};

/**
 * \brief Returns the access control level for the C++ base specifier
 *  represented by a cursor with kind CX_CXXBaseSpecifier.
 */
CINDEX_LINKAGE enum CX_CXXAccessSpecifier clang_getCXXAccessSpecifier(CXCursor);

/**
 * @}
 */
  
/**
 * \defgroup CINDEX_ATTRIBUTES Information for attributes
 *
 * @{
 */


/**
 * \brief For cursors representing an iboutletcollection attribute,
 *  this function returns the collection element type.
 *
 */
CINDEX_LINKAGE CXType clang_getIBOutletCollectionType(CXCursor);

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
 * \param parent the cursor whose child may be visited. All kinds of
 * cursors can be visited, including invalid cursors (which, by
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
 * \brief Construct a USR for a specified Objective-C class.
 */
CINDEX_LINKAGE CXString clang_constructUSR_ObjCClass(const char *class_name);

/**
 * \brief Construct a USR for a specified Objective-C category.
 */
CINDEX_LINKAGE CXString
  clang_constructUSR_ObjCCategory(const char *class_name,
                                 const char *category_name);

/**
 * \brief Construct a USR for a specified Objective-C protocol.
 */
CINDEX_LINKAGE CXString
  clang_constructUSR_ObjCProtocol(const char *protocol_name);


/**
 * \brief Construct a USR for a specified Objective-C instance variable and
 *   the USR for its containing class.
 */
CINDEX_LINKAGE CXString clang_constructUSR_ObjCIvar(const char *name,
                                                    CXString classUSR);

/**
 * \brief Construct a USR for a specified Objective-C method and
 *   the USR for its containing class.
 */
CINDEX_LINKAGE CXString clang_constructUSR_ObjCMethod(const char *name,
                                                      unsigned isInstanceMethod,
                                                      CXString classUSR);

/**
 * \brief Construct a USR for a specified Objective-C property and the USR
 *  for its containing class.
 */
CINDEX_LINKAGE CXString clang_constructUSR_ObjCProperty(const char *property,
                                                        CXString classUSR);

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
 * \defgroup CINDEX_CPP C++ AST introspection
 *
 * The routines in this group provide access information in the ASTs specific
 * to C++ language features.
 *
 * @{
 */

/**
 * \brief Determine if a C++ member function or member function template is 
 * declared 'static'.
 */
CINDEX_LINKAGE unsigned clang_CXXMethod_isStatic(CXCursor C);

/**
 * \brief Given a cursor that represents a template, determine
 * the cursor kind of the specializations would be generated by instantiating
 * the template.
 *
 * This routine can be used to determine what flavor of function template,
 * class template, or class template partial specialization is stored in the
 * cursor. For example, it can describe whether a class template cursor is
 * declared with "struct", "class" or "union".
 *
 * \param C The cursor to query. This cursor should represent a template
 * declaration.
 *
 * \returns The cursor kind of the specializations that would be generated
 * by instantiating the template \p C. If \p C is not a template, returns
 * \c CXCursor_NoDeclFound.
 */
CINDEX_LINKAGE enum CXCursorKind clang_getTemplateCursorKind(CXCursor C);
  
/**
 * @}
 */

/**
 * \defgroup CINDEX_LEX Token extraction and manipulation
 *
 * The routines in this group provide access to the tokens within a
 * translation unit, along with a semantic mapping of those tokens to
 * their corresponding cursors.
 *
 * @{
 */

/**
 * \brief Describes a kind of token.
 */
typedef enum CXTokenKind {
  /**
   * \brief A token that contains some kind of punctuation.
   */
  CXToken_Punctuation,

  /**
   * \brief A language keyword.
   */
  CXToken_Keyword,

  /**
   * \brief An identifier (that is not a keyword).
   */
  CXToken_Identifier,

  /**
   * \brief A numeric, string, or character literal.
   */
  CXToken_Literal,

  /**
   * \brief A comment.
   */
  CXToken_Comment
} CXTokenKind;

/**
 * \brief Describes a single preprocessing token.
 */
typedef struct {
  unsigned int_data[4];
  void *ptr_data;
} CXToken;

/**
 * \brief Determine the kind of the given token.
 */
CINDEX_LINKAGE CXTokenKind clang_getTokenKind(CXToken);

/**
 * \brief Determine the spelling of the given token.
 *
 * The spelling of a token is the textual representation of that token, e.g.,
 * the text of an identifier or keyword.
 */
CINDEX_LINKAGE CXString clang_getTokenSpelling(CXTranslationUnit, CXToken);

/**
 * \brief Retrieve the source location of the given token.
 */
CINDEX_LINKAGE CXSourceLocation clang_getTokenLocation(CXTranslationUnit,
                                                       CXToken);

/**
 * \brief Retrieve a source range that covers the given token.
 */
CINDEX_LINKAGE CXSourceRange clang_getTokenExtent(CXTranslationUnit, CXToken);

/**
 * \brief Tokenize the source code described by the given range into raw
 * lexical tokens.
 *
 * \param TU the translation unit whose text is being tokenized.
 *
 * \param Range the source range in which text should be tokenized. All of the
 * tokens produced by tokenization will fall within this source range,
 *
 * \param Tokens this pointer will be set to point to the array of tokens
 * that occur within the given source range. The returned pointer must be
 * freed with clang_disposeTokens() before the translation unit is destroyed.
 *
 * \param NumTokens will be set to the number of tokens in the \c *Tokens
 * array.
 *
 */
CINDEX_LINKAGE void clang_tokenize(CXTranslationUnit TU, CXSourceRange Range,
                                   CXToken **Tokens, unsigned *NumTokens);

/**
 * \brief Annotate the given set of tokens by providing cursors for each token
 * that can be mapped to a specific entity within the abstract syntax tree.
 *
 * This token-annotation routine is equivalent to invoking
 * clang_getCursor() for the source locations of each of the
 * tokens. The cursors provided are filtered, so that only those
 * cursors that have a direct correspondence to the token are
 * accepted. For example, given a function call \c f(x),
 * clang_getCursor() would provide the following cursors:
 *
 *   * when the cursor is over the 'f', a DeclRefExpr cursor referring to 'f'.
 *   * when the cursor is over the '(' or the ')', a CallExpr referring to 'f'.
 *   * when the cursor is over the 'x', a DeclRefExpr cursor referring to 'x'.
 *
 * Only the first and last of these cursors will occur within the
 * annotate, since the tokens "f" and "x' directly refer to a function
 * and a variable, respectively, but the parentheses are just a small
 * part of the full syntax of the function call expression, which is
 * not provided as an annotation.
 *
 * \param TU the translation unit that owns the given tokens.
 *
 * \param Tokens the set of tokens to annotate.
 *
 * \param NumTokens the number of tokens in \p Tokens.
 *
 * \param Cursors an array of \p NumTokens cursors, whose contents will be
 * replaced with the cursors corresponding to each token.
 */
CINDEX_LINKAGE void clang_annotateTokens(CXTranslationUnit TU,
                                         CXToken *Tokens, unsigned NumTokens,
                                         CXCursor *Cursors);

/**
 * \brief Free the given set of tokens.
 */
CINDEX_LINKAGE void clang_disposeTokens(CXTranslationUnit TU,
                                        CXToken *Tokens, unsigned NumTokens);

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
CINDEX_LINKAGE CXString clang_getCursorKindSpelling(enum CXCursorKind Kind);
CINDEX_LINKAGE void clang_getDefinitionSpellingAndExtent(CXCursor,
                                          const char **startBuf,
                                          const char **endBuf,
                                          unsigned *startLine,
                                          unsigned *startColumn,
                                          unsigned *endLine,
                                          unsigned *endColumn);
CINDEX_LINKAGE void clang_enableStackTraces(void);
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
   *       - a Placeholder chunk for "float y"
   *       - an Optional chunk containing the last defaulted argument:
   *           - a Comma chunk for ","
   *           - a Placeholder chunk for "double z"
   *   - a RightParen chunk for ")"
   *
   * There are many ways to handle Optional chunks. Two simple approaches are:
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
CINDEX_LINKAGE CXString
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
 * \brief Determine the priority of this code completion.
 *
 * The priority of a code completion indicates how likely it is that this 
 * particular completion is the completion that the user will select. The
 * priority is selected by various internal heuristics.
 *
 * \param completion_string The completion string to query.
 *
 * \returns The priority of this completion string. Smaller values indicate
 * higher-priority (more likely) completions.
 */
CINDEX_LINKAGE unsigned
clang_getCompletionPriority(CXCompletionString completion_string);
  
/**
 * \brief Determine the availability of the entity that this code-completion
 * string refers to.
 *
 * \param completion_string The completion string to query.
 *
 * \returns The availability of the completion string.
 */
CINDEX_LINKAGE enum CXAvailabilityKind 
clang_getCompletionAvailability(CXCompletionString completion_string);

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
 * those files.  The contents and name of these files (as specified by
 * CXUnsavedFile) are copied when necessary, so the client only needs to
 * guarantee their validity until the call to this function returns.
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
 * \param diag_callback callback function that will receive any diagnostics
 * emitted while processing this source file. If NULL, diagnostics will be
 * suppressed.
 *
 * \param diag_client_data client data that will be passed to the diagnostic
 * callback function.
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
                                          const char * const *command_line_args,
                                          unsigned num_unsaved_files,
                                          struct CXUnsavedFile *unsaved_files,
                                          const char *complete_filename,
                                          unsigned complete_line,
                                          unsigned complete_column);

/**
 * \brief Flags that can be passed to \c clang_codeCompleteAt() to
 * modify its behavior.
 *
 * The enumerators in this enumeration can be bitwise-OR'd together to
 * provide multiple options to \c clang_codeCompleteAt().
 */
enum CXCodeComplete_Flags {
  /**
   * \brief Whether to include macros within the set of code
   * completions returned.
   */
  CXCodeComplete_IncludeMacros = 0x01,

  /**
   * \brief Whether to include code patterns for language constructs
   * within the set of code completions, e.g., for loops.
   */
  CXCodeComplete_IncludeCodePatterns = 0x02
};

/**
 * \brief Returns a default set of code-completion options that can be
 * passed to\c clang_codeCompleteAt(). 
 */
CINDEX_LINKAGE unsigned clang_defaultCodeCompleteOptions(void);

/**
 * \brief Perform code completion at a given location in a translation unit.
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
 * \param TU The translation unit in which code-completion should
 * occur. The source files for this translation unit need not be
 * completely up-to-date (and the contents of those source files may
 * be overridden via \p unsaved_files). Cursors referring into the
 * translation unit may be invalidated by this invocation.
 *
 * \param complete_filename The name of the source file where code
 * completion should be performed. This filename may be any file
 * included in the translation unit.
 *
 * \param complete_line The line at which code-completion should occur.
 *
 * \param complete_column The column at which code-completion should occur.
 * Note that the column should point just after the syntactic construct that
 * initiated code completion, and not in the middle of a lexical token.
 *
 * \param unsaved_files the Tiles that have not yet been saved to disk
 * but may be required for parsing or code completion, including the
 * contents of those files.  The contents and name of these files (as
 * specified by CXUnsavedFile) are copied when necessary, so the
 * client only needs to guarantee their validity until the call to
 * this function returns.
 *
 * \param num_unsaved_files The number of unsaved file entries in \p
 * unsaved_files.
 *
 * \param options Extra options that control the behavior of code
 * completion, expressed as a bitwise OR of the enumerators of the
 * CXCodeComplete_Flags enumeration. The 
 * \c clang_defaultCodeCompleteOptions() function returns a default set
 * of code-completion options.
 *
 * \returns If successful, a new \c CXCodeCompleteResults structure
 * containing code-completion results, which should eventually be
 * freed with \c clang_disposeCodeCompleteResults(). If code
 * completion fails, returns NULL.
 */
CINDEX_LINKAGE
CXCodeCompleteResults *clang_codeCompleteAt(CXTranslationUnit TU,
                                            const char *complete_filename,
                                            unsigned complete_line,
                                            unsigned complete_column,
                                            struct CXUnsavedFile *unsaved_files,
                                            unsigned num_unsaved_files,
                                            unsigned options);

/**
 * \brief Sort the code-completion results in case-insensitive alphabetical 
 * order.
 *
 * \param Results The set of results to sort.
 * \param NumResults The number of results in \p Results.
 */
CINDEX_LINKAGE
void clang_sortCodeCompletionResults(CXCompletionResult *Results,
                                     unsigned NumResults);
  
/**
 * \brief Free the given set of code-completion results.
 */
CINDEX_LINKAGE
void clang_disposeCodeCompleteResults(CXCodeCompleteResults *Results);
  
/**
 * \brief Determine the number of diagnostics produced prior to the
 * location where code completion was performed.
 */
CINDEX_LINKAGE
unsigned clang_codeCompleteGetNumDiagnostics(CXCodeCompleteResults *Results);

/**
 * \brief Retrieve a diagnostic associated with the given code completion.
 *
 * \param Result the code completion results to query.
 * \param Index the zero-based diagnostic number to retrieve.
 *
 * \returns the requested diagnostic. This diagnostic must be freed
 * via a call to \c clang_disposeDiagnostic().
 */
CINDEX_LINKAGE
CXDiagnostic clang_codeCompleteGetDiagnostic(CXCodeCompleteResults *Results,
                                             unsigned Index);

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
CINDEX_LINKAGE CXString clang_getClangVersion();

/**
 * \brief Return a version string, suitable for showing to a user, but not
 *        intended to be parsed (the format is not guaranteed to be stable).
 */


 /**
  * \brief Visitor invoked for each file in a translation unit
  *        (used with clang_getInclusions()).
  *
  * This visitor function will be invoked by clang_getInclusions() for each
  * file included (either at the top-level or by #include directives) within
  * a translation unit.  The first argument is the file being included, and
  * the second and third arguments provide the inclusion stack.  The
  * array is sorted in order of immediate inclusion.  For example,
  * the first element refers to the location that included 'included_file'.
  */
typedef void (*CXInclusionVisitor)(CXFile included_file,
                                   CXSourceLocation* inclusion_stack,
                                   unsigned include_len,
                                   CXClientData client_data);

/**
 * \brief Visit the set of preprocessor inclusions in a translation unit.
 *   The visitor function is called with the provided data for every included
 *   file.  This does not include headers included by the PCH file (unless one
 *   is inspecting the inclusions in the PCH file itself).
 */
CINDEX_LINKAGE void clang_getInclusions(CXTranslationUnit tu,
                                        CXInclusionVisitor visitor,
                                        CXClientData client_data);

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

