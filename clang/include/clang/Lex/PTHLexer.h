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
  unsigned LastToken;
  
  /// CurToken - This is the index of the next token that Lex will return.
  unsigned CurToken;
        
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
  unsigned isNextPPTokenLParen();
  
  /// IndirectLex - An indirect call to 'Lex' that can be invoked via
  ///  the PreprocessorLexer interface.
  void IndirectLex(Token &Result) { Lex(Result); }
};

}  // end namespace clang

#endif
