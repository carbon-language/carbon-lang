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
                 'whole program' ASTs depending on how the use the APIs.
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
 

II. Current advantages over GCC:

 * Column numbers are fully tracked (no 256 col limit, no GCC-style pruning).
 * All diagnostics have column numbers, includes 'caret diagnostics'.
 * Full diagnostic customization by client (can format diagnostics however they
   like, e.g. in an IDE or refactoring tool).
 * Built as a framework, can be reused by multiple tools.
 * All languages supported linked into same library (no cc1,cc1obj, ...).
 * mmap's code in read-only, does not dirty the pages like GCC (mem footprint).
 * BSD License, can be linked into non-GPL projects.
 * Full diagnostic control, per diagnostic.
 * Faster than GCC at parsing, lexing, and preprocessing.
 
Future Features:

 * Fine grained diag control within the source (#pragma enable/disable warning).
 * Faster than GCC at IR generation [measure when complete].
 * Better token tracking within macros?  (Token came from this line, which is
   a macro argument instantiated here, recursively instantiated here).
 * Fast #import!
 * Dependency tracking: change to header file doesn't recompile every function
   that texually depends on it: recompile only those functions that need it.
 * Defers exposing platform-specific stuff to as late as possible, tracks use of
   platform-specific features (e.g. #ifdef PPC) to allow 'portable bytecodes'.


III. Missing Functionality / Improvements

File Manager:
 * We currently do a lot of stat'ing for files that don't exist, particularly
   when lots of -I paths exist (e.g. see the <iostream> example, check for
   failures in stat in FileManager::getFile).  It would be far better to make
   the following changes:
     1. FileEntry contains a sys::Path instead of a std::string for Name.
     2. sys::Path contains timestamp and size, lazily computed.  Eliminate from
        FileEntry.
     3. File UIDs are created on request, not when files are opened.
   These changes make it possible to efficiently have FileEntry objects for
   files that exist on the file system, but have not been used yet.
   
   Once this is done:
     1. DirectoryEntry gets a boolean value "has read entries".  When false, not
        all entries in the directory are in the file mgr, when true, they are.
     2. Instead of stat'ing the file in FileManager::getFile, check to see if 
        the dir has been read.  If so, fail immediately, if not, read the dir,
        then retry.
     3. Reading the dir uses the getdirentries syscall, creating an FileEntry
        for all files found.

Lexer:
 * Source character mapping.  GCC supports ASCII and UTF-8.
   See GCC options: -ftarget-charset and -ftarget-wide-charset.
 * Universal character support.  Experimental in GCC, enabled with
   -fextended-identifiers.
 * -fpreprocessed mode.

Preprocessor:
 * Know enough about darwin filesystem to search frameworks.
 * #assert/#unassert
 * #line / #file directives
 * MSExtension: "L#param" stringizes to a wide string literal.

Traditional Preprocessor:
 * All.

Parser:
 * C90/K&R modes.  Need to get a copy of the C90 spec.
 * __extension__, __attribute__ [currently just skipped and ignored].

Parser Callbacks:
 * Enough to do devkit-style "indexing".
 * All that are missing.
 
Parser Actions:
 * All.
 * Would like to either lazily resolve types [refactoring] or aggressively
   resolve them [c compiler].  Need to know whether something is a type or not
   to compile, but don't need to know what it is.

Fast #Import:
 * All.
 * Get frameworks that don't use #import to do so, e.g. 
   DirectoryService, AudioToolbox, CoreFoundation, etc.  Why not using #import?
   Because they work in C mode? C has #import.
 * Have the lexer return a token for #import instead of handling it itself.
   - Create a new preprocessor object with no external state (no -D/U options
     from the command line, etc).  Alternatively, keep track of exactly which
     external state is used by a #import: declare it somehow.
 * When having reading a #import file, keep track of whether we have (and/or
   which) seen any "configuration" macros.  Various cases:
   - Uses of target args (__POWERPC__, __i386): Header has to be parsed 
     multiple times, per-target.  What about #ifndef checks?  How do we know?
   - "Configuration" preprocessor macros not defined: POWERPC, etc.  What about
     things like __STDC__ etc?  What is and what isn't allowed.
 * Special handling for "umbrella" headers, which just contain #import stmts:
   - Cocoa.h/AppKit.h - Contain pointers to digests instead of entire digests
     themselves?  Foundation.h isn't pure umbrella!
 * Frameworks digests:
   - Can put "digest" of a framework-worth of headers into the framework
     itself.  To open AppKit, just mmap
     /System/Library/Frameworks/AppKit.framework/"digest", which provides a
     symbol table in a well defined format.  Lazily unstream stuff that is
     needed.  Contains declarations, macros, and debug information.
   - System frameworks ship with digests.  How do we handle configuration
     information?  How do we handle stuff like:
       #if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_2
     which guards a bunch of decls?  Should there be a couple of default
     configs, then have the UI fall back to building/caching its own?
   - GUI automatically builds digests when UI is idle, both of system
     frameworks if they aren't not available in the right config, and of app
     frameworks.
   - GUI builds dependence graph of frameworks/digests based on #imports.  If a
     digest is out date, dependent digests are automatically invalidated.

 * New constraints on #import for objc-v3:
   - #imported file must not define non-inline function bodies.
     - Alternatively, they can, and these bodies get compiled/linked *once*
       per app into a dylib.  What about building user dylibs?
   - Restrictions on ObjC grammar: can't #import the body of a for stmt or fn.
   - Compiler must detect and reject these cases.
   - #defines defined within a #import have two behaviors:
     - By default, they escape the header.  These macros *cannot* be #undef'd
       by other code: this is enforced by the front-end.
     - Optionally, user can specify what macros escape (whitelist) or can use
       #undef.

New language feature: Configuration queries:
  - Instead of #ifdef __POWERPC__, use "if (strcmp(`cpu`, __POWERPC__))", or
    some other syntax.
  - Use it to increase the number of "architecture-clean" #import'd files,
    allowing a single index to be used for all fat slices.

Cocoa GUI Front-end:
 * All.
 * Start with very simple "textedit" GUI.
 * Trivial project model: list of files, list of cmd line options.
 * Build simple developer examples.
 * Tight integration with compiler components.
 * Primary advantage: batch compiles, keeping digests in memory, dependency mgmt
   between app frameworks, building code/digests in the background, etc.
 * Interesting idea: http://nickgravgaard.com/elastictabstops/
 
