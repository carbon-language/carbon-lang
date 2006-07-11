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
#include <vector>

namespace llvm {
namespace clang {
  class MacroInfo;
  class Preprocessor;
  class LexerToken;

/// MacroFormalArgs - An instance of this class captures information about
/// the formal arguments specified to a function-like macro invocation.
class MacroFormalArgs {
  std::vector<std::vector<LexerToken> > ArgTokens;
public:
  MacroFormalArgs(const MacroInfo *MI);
  
  /// addArgument - Add an argument for this invocation.  This method destroys
  /// the vector passed in to avoid extraneous memory copies.
  void addArgument(std::vector<LexerToken> &ArgToks) {
    ArgTokens.push_back(std::vector<LexerToken>());
    ArgTokens.back().swap(ArgToks);
  }
  
  /// getNumArguments - Return the number of arguments passed into this macro
  /// invocation.
  unsigned getNumArguments() const { return ArgTokens.size(); }
};

  
/// MacroExpander - This implements a lexer that returns token from a macro body
/// instead of lexing from a character buffer.
///
class MacroExpander {
  /// Macro - The macro we are expanding from.
  ///
  MacroInfo &Macro;
  
  /// FormalArgs - The formal arguments specified for a function-like macro, or
  /// null.  The MacroExpander owns the pointed-to object.
  MacroFormalArgs *FormalArgs;

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
  
  MacroExpander(const MacroExpander&);  // DO NOT IMPLEMENT
  void operator=(const MacroExpander&); // DO NOT IMPLEMENT
public:
  /// Create a macro expander of the specified macro with the specified formal
  /// arguments.  Note that this ctor takes ownership of the FormalArgs pointer.
  MacroExpander(LexerToken &Tok, MacroFormalArgs *FormalArgs,
                Preprocessor &pp);
  ~MacroExpander() {
    // MacroExpander owns its formal arguments.
    delete FormalArgs;
  }
  
  /// NextTokenIsKnownNotLParen - If the next token lexed will pop this macro
  /// off the expansion stack, return false and set RanOffEnd to true.
  /// Otherwise, return true if we know for sure that the next token returned
  /// will not be a '(' token.  Return false if it is a '(' token or if we are
  /// not sure.  This is used when determining whether to expand a function-like
  /// macro.
  bool NextTokenIsKnownNotLParen(bool &RanOffEnd) const;
  
  MacroInfo &getMacro() const { return Macro; }

  /// Lex - Lex and return a token from this macro stream.
  void Lex(LexerToken &Tok);
};

}  // end namespace llvm
}  // end namespace clang

#endif
