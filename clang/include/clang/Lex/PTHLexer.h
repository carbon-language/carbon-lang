//===--- PTHLexer.h - Lexer based on Pre-tokenized input --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PTHLexer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PTHLEXER_H
#define LLVM_CLANG_PTHLEXER_H

#include "clang/Lex/PreprocessorLexer.h"

namespace clang {
  
class PTHLexer : public PreprocessorLexer {
  /// Tokens - This is the pointer to an array of tokens that the macro is
  /// defined to, with arguments expanded for function-like macros.  If this is
  /// a token stream, these are the tokens we are returning.
  const Token *Tokens;
  
  /// LastTokenIdx - The index of the last token in Tokens.  This token
  ///  will be an eof token.
  unsigned LastTokenIdx;
  
  /// CurTokenIdx - This is the index of the next token that Lex will return.
  unsigned CurTokenIdx;
        
  PTHLexer(const PTHLexer&);  // DO NOT IMPLEMENT
  void operator=(const PTHLexer&); // DO NOT IMPLEMENT

public:

  /// Create a PTHLexer for the specified token stream.
  PTHLexer(Preprocessor& pp, SourceLocation fileloc,
           const Token *TokArray, unsigned NumToks);
  ~PTHLexer() {}
    
  /// Lex - Return the next token.
  void Lex(Token &Tok);
  
  void setEOF(Token &Tok);
  
  /// DiscardToEndOfLine - Read the rest of the current preprocessor line as an
  /// uninterpreted string.  This switches the lexer out of directive mode.
  void DiscardToEndOfLine();
  
  /// isNextPPTokenLParen - Return 1 if the next unexpanded token will return a
  /// tok::l_paren token, 0 if it is something else and 2 if there are no more
  /// tokens controlled by this lexer.
  unsigned isNextPPTokenLParen() {
    return AtLastToken() ? 2 : GetToken().is(tok::l_paren);
  }

  /// IndirectLex - An indirect call to 'Lex' that can be invoked via
  ///  the PreprocessorLexer interface.
  void IndirectLex(Token &Result) { Lex(Result); }
  
private:
  
  /// AtLastToken - Returns true if the PTHLexer is at the last token.
  bool AtLastToken() const { return CurTokenIdx == LastTokenIdx; }
  
  /// GetToken - Returns the next token.  This method does not advance the
  ///  PTHLexer to the next token.
  Token GetToken() { return Tokens[CurTokenIdx]; }
  
  /// AdvanceToken - Advances the PTHLexer to the next token.
  void AdvanceToken() { ++CurTokenIdx; }
};

}  // end namespace clang

#endif
