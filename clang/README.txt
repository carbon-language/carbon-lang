//===----------------------------------------------------------------------===//
// C Language Family Front-end
//===----------------------------------------------------------------------===//
                                                             Chris Lattner

I. Introduction:
 
 clang: noun
    1. A loud, resonant, metallic sound.
    2. The strident call of a crane or goose.
    3. C-language front-end toolkit.

 The world needs better compiler tools, tools which are built as libraries. This
 design point allows reuse of the tools in new and novel ways. However, building
 the tools as libraries isn't enough: they must have clean APIs, be as
 decoupled from each other as possible, and be easy to modify/extend.  This
 requires clean layering, decent design, and avoiding tying the libraries to a
 specific use.  Oh yeah, did I mention that we want the resultant libraries to
 be as fast as possible? :)

 This front-end is built as a component of the LLVM toolkit (which really really
 needs a better name) that can be used with the LLVM backend or independently of
 it.  In this spirit, the API has been carefully designed to include the
 following components:
 
   libsupport  - Basic support library, reused from LLVM.
   libsystem   - System abstraction library, reused from LLVM.
   libbasic    - Diagnostics, SourceLocations, SourceBuffer abstraction,
                 file system caching for input source files.
   liblex      - C/C++/ObjC lexing and preprocessing, identifier hash table,
                 pragma handling, tokens, and macros.
   libparse    - C99 (for now) parsing and local semantic analysis. This library
                 invokes coarse-grained 'Actions' provided by the client to do
                 stuff (great idea shamelessly stolen from Devkit).  ObjC/C90
                 need to be added soon, K&R C and C++ can be added in the
                 future, but are not a high priority.
   libast      - Provides a set of parser actions to build a standardized AST
                 for programs.  AST can be built in two forms: streamlined and
                 'complete' mode, which captures *full* location info for every
                 token in the AST.  AST's are 'streamed' out a top-level
                 declaration at a time, allowing clients to use decl-at-a-time
                 processing, build up entire translation units, or even build
                 'whole program' ASTs depending on how they use the APIs.
   libast2llvm - [Planned] Lower the AST to LLVM IR for optimization & codegen.
   clang       - An example client of the libraries at various levels.

 This front-end has been intentionally built as a stack, making it trivial
 to replace anything below a particular point.  For example, if you want a
 preprocessor, you take the Basic and Lexer libraries.  If you want an indexer,
 you take those plus the Parser library and provide some actions for indexing.
 If you want a refactoring, static analysis, or source-to-source compiler tool,
 it makes sense to take those plus the AST building library.  Finally, if you
 want to use this with the LLVM backend, you'd take these components plus the
 AST to LLVM lowering code.
 
 In the future I hope this toolkit will grow to include new and interesting
 components, including a C++ front-end, ObjC support, AST pretty printing
 support, and a whole lot of other things.

 Finally, it should be pointed out that the goal here is to build something that
 is high-quality and industrial-strength: all the obnoxious features of the C
 family must be correctly supported (trigraphs, preprocessor arcana, K&R-style
 prototypes, GCC/MS extensions, etc).  It cannot be used if it's not 'real'.


II. Usage of clang driver:

 * Basic Command-Line Options:
   - Help: clang --help
   - Standard GCC options accepted: -E, -I*, -i*, -pedantic, -std=c90, etc.
   - Make diagnostics more gcc-like: -fno-caret-diagnostics -fno-show-column
   - Enable metric printing: -stats

 * -parse-noop is the default mode.

 * -E mode gives output nearly identical to GCC, though not all bugs in
   whitespace calculation have been emulated.

 * -fsyntax-only is currently unimplemented.
 
 * -parse-print-callbacks prints almost no callbacks so far.
 
 * -parse-ast builds ASTs, but doesn't print them.  This is most useful for
   timing AST building vs -parse-noop.
 
 * -parse-ast-print prints most expression and statements nodes, but some
   things are missing.

III. Current advantages over GCC:

 * Column numbers are fully tracked (no 256 col limit, no GCC-style pruning).
 * All diagnostics have column numbers, includes 'caret diagnostics'.
 * Full diagnostic customization by client (can format diagnostics however they
   like, e.g. in an IDE or refactoring tool) through DiagnosticClient interface.
 * Built as a framework, can be reused by multiple tools.
 * All languages supported linked into same library (no cc1,cc1obj, ...).
 * mmap's code in read-only, does not dirty the pages like GCC (mem footprint).
 * BSD License, can be linked into non-GPL projects.
 * Full diagnostic control, per diagnostic.
 * Faster than GCC at parsing, lexing, and preprocessing.
 * Defers exposing platform-specific stuff to as late as possible, tracks use of
   platform-specific features (e.g. #ifdef PPC) to allow 'portable bytecodes'.

Future Features:

 * Fine grained diag control within the source (#pragma enable/disable warning).
 * Faster than GCC at AST generation [measure when complete].
 * Better token tracking within macros?  (Token came from this line, which is
   a macro argument instantiated here, recursively instantiated here).
 * Fast #import!
 * Dependency tracking: change to header file doesn't recompile every function
   that texually depends on it: recompile only those functions that need it.


IV. Missing Functionality / Improvements

clang driver:
 * Include search paths are hard-coded into the driver.

File Manager:
 * Reduce syscalls, see NOTES.txt.

Lexer:
 * Source character mapping.  GCC supports ASCII and UTF-8.
   See GCC options: -ftarget-charset and -ftarget-wide-charset.
 * Universal character support.  Experimental in GCC, enabled with
   -fextended-identifiers.
 * -fpreprocessed mode.

Preprocessor:
 * Know about apple header maps.
 * #assert/#unassert
 * #line / #file directives (currently accepted and ignored).
 * MSExtension: "L#param" stringizes to a wide string literal.
 * Consider merging the parser's expression parser into the preprocessor to
   eliminate duplicate code.
 * Add support for -M*

Traditional Preprocessor:
 * All.

Parser:
 * C90/K&R modes.  Need to get a copy of the C90 spec.
 * __extension__, __attribute__ [currently just skipped and ignored].
 * A lot of semantic analysis is missing.
 * "initializers", GCC inline asm.

Parser Actions:
 * All that are missing.
 * SemaActions vs MinimalActions.
 * Would like to either lazily resolve types [refactoring] or aggressively
   resolve them [c compiler].  Need to know whether something is a type or not
   to compile, but don't need to know what it is.
 * Implement a little devkit-style "indexer".
 
AST Builder:
 * Implement more nodes as actions are available.
 * Types.
 * Decls.
