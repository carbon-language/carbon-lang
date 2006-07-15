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
  
  /// StringifiedArgs - This contains arguments in 'stringified' form.  If the
  /// stringified form of an argument has not yet been computed, this is empty.
  std::vector<LexerToken> StringifiedArgs;
public:
  MacroFormalArgs(const MacroInfo *MI);
  
  /// addArgument - Add an argument for this invocation.  This method destroys
  /// the vector passed in to avoid extraneous memory copies.
  void addArgument(std::vector<LexerToken> &ArgToks) {
    ArgTokens.push_back(std::vector<LexerToken>());
    ArgTokens.back().swap(ArgToks);
  }
  
  /// getUnexpArgument - Return the unexpanded tokens for the specified formal.
  ///
  const std::vector<LexerToken> &getUnexpArgument(unsigned Arg) const {
    assert(Arg < ArgTokens.size() && "Invalid ArgNo");
    return ArgTokens[Arg];
  }
  
  /// getStringifiedArgument - Compute, cache, and return the specified argument
  /// that has been 'stringified' as required by the # operator.
  const LexerToken &getStringifiedArgument(unsigned ArgNo, Preprocessor &PP);
  
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

  /// MacroTokens - This is the pointer to the list of tokens that the macro is
  /// defined to, with arguments expanded for function-like macros.
  const std::vector<LexerToken> *MacroTokens;
  
  /// CurToken - This is the next token that Lex will return.
  ///
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
  ~MacroExpander();
  
  /// isNextTokenLParen - If the next token lexed will pop this macro off the
  /// expansion stack, return 2.  If the next unexpanded token is a '(', return
  /// 1, otherwise return 0.
  unsigned isNextTokenLParen() const;
  
  MacroInfo &getMacro() const { return Macro; }

  /// Lex - Lex and return a token from this macro stream.
  void Lex(LexerToken &Tok);
  
private:
  /// isAtEnd - Return true if the next lex call will pop this macro off the
  /// include stack.
  bool isAtEnd() const {
    return CurToken == MacroTokens->size();
  }
  
  /// Expand the arguments of a function-like macro so that we can quickly
  /// return preexpanded tokens from MacroTokens.
  void ExpandFunctionArguments();
};

}  // end namespace llvm
}  // end namespace clang

#endif
