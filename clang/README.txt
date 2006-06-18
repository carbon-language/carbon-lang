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

 * Full column number support in diagnostics.
 * Caret diagnostics.
 * Full diagnostic customization by client (can format diagnostics however they
   like, e.g. in an IDE or refactoring tool).
 * Built as a framework, can be reused by multiple tools.
 * All languages supported linked into same library (no cc1,cc1obj, ...).
 * mmap's code in read-only, does not dirty the pages like GCC (mem footprint).
 * BSD License, can be linked into non-GPL projects.
 
Future Features:
 * Full diagnostic control, per diagnostic (use enums).
 * Fine grained control within the source (#pragma enable/disable warning)
 * Faster than GCC, preprocessing, parsing, IR generation.
 * Better token tracking within macros?  (Token came from this line, which is
   a macro argument instantiated here, recursively instantiated here).
 * Fast #import!!


III. Critical Missing Functionality

Lexer:
 * Source character mapping.  GCC supports ASCII and UTF-8.
   See GCC options: -ftarget-charset and -ftarget-wide-charset.
 * Universal character support.  Experimental in GCC, enabled with
   -fextended-identifiers.
 * Poisoned identifiers.
 * -fpreprocessed mode.

Preprocessor:
 * #line / #file directives
 * Detection of "atomic" headers (#ifndef/#define), #pragma once support.
 * Function-style #define & macro expansion
 * -C & -P output modes.

Traditional Preprocessor:
 * All.
    
Parser Callbacks:
 * All.
 
Parser Actions:
 * All.
 