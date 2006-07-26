//===--- MacroExpander.h - Lex from a macro expansion -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MacroExpander and MacroArgs interfaces.
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

/// MacroArgs - An instance of this class captures information about
/// the formal arguments specified to a function-like macro invocation.
class MacroArgs {
  /// UnexpArgTokens - Raw, unexpanded tokens for the arguments.  This is all of
  /// the arguments concatenated together, with 'EOF' markers at the end of each
  /// argument.
  std::vector<LexerToken> UnexpArgTokens;

  /// PreExpArgTokens - Pre-expanded tokens for arguments that need them.  Empty
  /// if not yet computed.  This includes the EOF marker at the end of the
  /// stream.
  std::vector<std::vector<LexerToken> > PreExpArgTokens;

  /// StringifiedArgs - This contains arguments in 'stringified' form.  If the
  /// stringified form of an argument has not yet been computed, this is empty.
  std::vector<LexerToken> StringifiedArgs;
public:
  /// MacroArgs ctor - This destroys the vector passed in.
  MacroArgs(const MacroInfo *MI, std::vector<LexerToken> &UnexpArgTokens);
  
  
  /// ArgNeedsPreexpansion - If we can prove that the argument won't be affected
  /// by pre-expansion, return false.  Otherwise, conservatively return true.
  bool ArgNeedsPreexpansion(const LexerToken *ArgTok) const;
  
  /// getUnexpArgument - Return a pointer to the first token of the unexpanded
  /// token list for the specified formal.
  ///
  const LexerToken *getUnexpArgument(unsigned Arg) const;
  
  /// getArgLength - Given a pointer to an expanded or unexpanded argument,
  /// return the number of tokens, not counting the EOF, that make up the
  /// argument.
  static unsigned getArgLength(const LexerToken *ArgPtr);
  
  /// getPreExpArgument - Return the pre-expanded form of the specified
  /// argument.
  const std::vector<LexerToken> &
    getPreExpArgument(unsigned Arg, Preprocessor &PP);  
  
  /// getStringifiedArgument - Compute, cache, and return the specified argument
  /// that has been 'stringified' as required by the # operator.
  const LexerToken &getStringifiedArgument(unsigned ArgNo, Preprocessor &PP);
  
  /// getNumArguments - Return the number of arguments passed into this macro
  /// invocation.
  unsigned getNumArguments() const { return UnexpArgTokens.size(); }
};

  
/// MacroExpander - This implements a lexer that returns token from a macro body
/// or token stream instead of lexing from a character buffer.
///
class MacroExpander {
  /// Macro - The macro we are expanding from.  This is null if expanding a
  /// token stream.
  ///
  MacroInfo *Macro;

  /// ActualArgs - The actual arguments specified for a function-like macro, or
  /// null.  The MacroExpander owns the pointed-to object.
  MacroArgs *ActualArgs;

  /// PP - The current preprocessor object we are expanding for.
  ///
  Preprocessor &PP;

  /// MacroTokens - This is the pointer to an array of tokens that the macro is
  /// defined to, with arguments expanded for function-like macros.  If this is
  /// a token stream, these are the tokens we are returning.
  const LexerToken *MacroTokens;
  
  /// NumMacroTokens - This is the length of the MacroTokens array.
  ///
  unsigned NumMacroTokens;
  
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
  /// Create a macro expander for the specified macro with the specified actual
  /// arguments.  Note that this ctor takes ownership of the ActualArgs pointer.
  MacroExpander(LexerToken &Tok, MacroArgs *ActualArgs, Preprocessor &PP);
  
  /// Create a macro expander for the specified token stream.  This does not
  /// take ownership of the specified token vector.
  MacroExpander(const LexerToken *TokArray, unsigned NumToks, Preprocessor &PP);
  ~MacroExpander();
  
  /// isNextTokenLParen - If the next token lexed will pop this macro off the
  /// expansion stack, return 2.  If the next unexpanded token is a '(', return
  /// 1, otherwise return 0.
  unsigned isNextTokenLParen() const;
  
  /// Lex - Lex and return a token from this macro stream.
  void Lex(LexerToken &Tok);
  
private:
  /// isAtEnd - Return true if the next lex call will pop this macro off the
  /// include stack.
  bool isAtEnd() const {
    return CurToken == NumMacroTokens;
  }
  
  /// PasteTokens - Tok is the LHS of a ## operator, and CurToken is the ##
  /// operator.  Read the ## and RHS, and paste the LHS/RHS together.  If there
  /// are is another ## after it, chomp it iteratively.  Return the result as
  /// Tok.
  void PasteTokens(LexerToken &Tok);
  
  /// Expand the arguments of a function-like macro so that we can quickly
  /// return preexpanded tokens from MacroTokens.
  void ExpandFunctionArguments();
};

}  // end namespace llvm
}  // end namespace clang

#endif
