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

#ifndef LLVM_CLANG_PTHLexer_H
#define LLVM_CLANG_PTHLexer_H

#include "clang/Lex/PreprocessorLexer.h"

namespace clang {
  
class PTHLexer : public PreprocessorLexer {
  /// FileLoc - Location for the start of the file.
  ///
  SourceLocation FileLoc;
  
  /// Tokens - This is the pointer to an array of tokens that the macro is
  /// defined to, with arguments expanded for function-like macros.  If this is
  /// a token stream, these are the tokens we are returning.
  const Token *Tokens;
  
  /// NumTokens - This is the length of the Tokens array.
  ///
  unsigned NumTokens;
  
  /// CurToken - This is the next token that Lex will return.
  ///
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
  
  /// getFileLoc - Return the File Location for the file we are lexing out of.
  /// The physical location encodes the location where the characters come from,
  /// the virtual location encodes where we should *claim* the characters came
  /// from.  Currently this is only used by _Pragma handling.
  SourceLocation getFileLoc() const { return FileLoc; }
};

}  // end namespace clang

#endif
