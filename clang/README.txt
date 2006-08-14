//===----------------------------------------------------------------------===//
// C Language Family Front-end
//===----------------------------------------------------------------------===//

I. Introduction:
 
 clang: noun
    1. A loud, resonant, metallic sound.
    2. The strident call of a crane or goose.
    3. C-language front-end toolkit.
    
 Why?
 Supports Objective-C.


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
 * Faster than GCC at lexing and preprocessing.
 
Future Features:

 * Fine grained diag control within the source (#pragma enable/disable warning).
 * Faster than GCC at parsing, IR generation.
 * Better token tracking within macros?  (Token came from this line, which is
   a macro argument instantiated here, recursively instantiated here).
 * Fast #import!
 * Dependency tracking: change to header file doesn't recompile every function
   that texually depends on it: only recompile those that need to change.
 * Defers exposing platform-specific stuff to as late as possible, tracks use of
   platform-specific features (e.g. #ifdef PPC).


III. Missing Functionality

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
 * C90/K&R modes.  Need to get C90 spec.
 * __extension__
 * __attribute__

Parser Callbacks:
 * Enough to do devkit-style "indexing".
 * All.
 
Parser Actions:
 * All.
 * Need some way to effeciently either work in 'callback'/devkit mode or in
   default AST building mode.
 * Would like to either lazily resolve types [refactoring] or aggressively
   resolve them [c compiler].  Need to know whether something is a type or not
   to compile, but don't need to know what it is.

Fast #Import:
 * All.
 * Get frameworks that don't use #import to do so, e.g. 
   DirectoryService, AudioToolbox, CoreFoundation, etc.  Why not using #import,
   because they work in C mode?
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
  - Use it to increase the number of "architecture-clean" #import'd files.

Cocoa GUI Front-end:
 * All.
 * Start with very simple "textedit" GUI.
 * Trivial project model: list of files, list of cmd line options.
 * Build simple developer examples.
 * Tight integration with compiler components.
 * Primary advantage: batch compiles, keeping digests in memory, dependency mgmt
   between app frameworks, building code/digests in the background, etc.
 * Interesting idea: http://nickgravgaard.com/elastictabstops/
 
