//===--- MacroExpander.h - Lex from a macro expansion -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MacroExpander interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_MACROEXPANDER_H
#define LLVM_CLANG_MACROEXPANDER_H

#include "clang/Basic/SourceLocation.h"

namespace llvm {
namespace clang {
  class MacroInfo;
  class Preprocessor;
  class LexerToken;

/// MacroExpander - This implements a lexer that returns token from a macro body
/// instead of lexing from a character buffer.
///
class MacroExpander {
  /// Macro - The macro we are expanding from.
  ///
  MacroInfo &Macro;

  /// PP - The current preprocessor object we are expanding for.
  ///
  Preprocessor &PP;
  
  /// CurToken - This is the next token that Lex will return.
  unsigned CurToken;
  
  /// InstantiateLoc - The source location where this macro was instantiated.
  ///
  SourceLocation InstantiateLoc;
  
  /// Lexical information about the expansion point of the macro: the identifier
  /// that the macro expanded from had these properties.
  bool AtStartOfLine, HasLeadingSpace;
  
public:
  MacroExpander(LexerToken &Tok, Preprocessor &pp);
  
  MacroInfo &getMacro() const { return Macro; }

  /// getInstantiationLoc - Return a SourceLocation that specifies a macro
  /// instantiation whose physical location is PhysLoc but the logical location
  /// is InstantiationLoc.
  static SourceLocation getInstantiationLoc(Preprocessor &PP,
                                            SourceLocation PhysLoc,
                                            SourceLocation InstantiationLoc);
  
  /// Lex - Lex and return a token from this macro stream.
  void Lex(LexerToken &Tok);
};
  
}  // end namespace llvm
}  // end namespace clang

#endif
