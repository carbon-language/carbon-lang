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
#include <vector>

namespace clang {
  
class PTHManager;
  
class PTHLexer : public PreprocessorLexer {
  /// TokBuf - Buffer from PTH file containing raw token data.
  const char* TokBuf;
  
  /// CurPtr - Pointer into current offset of the token buffer where
  ///  the next token will be read.
  const char* CurPtr;
    
  /// LastHashTokPtr - Pointer into TokBuf of the last processed '#'
  ///  token that appears at the start of a line.
  const char* LastHashTokPtr;
  
  /// PPCond - Pointer to a side table in the PTH file that provides a
  ///  a consise summary of the preproccessor conditional block structure.
  ///  This is used to perform quick skipping of conditional blocks.
  const char* PPCond;
  
  /// CurPPCondPtr - Pointer inside PPCond that refers to the next entry
  ///  to process when doing quick skipping of preprocessor blocks.
  const char* CurPPCondPtr;

  PTHLexer(const PTHLexer&);  // DO NOT IMPLEMENT
  void operator=(const PTHLexer&); // DO NOT IMPLEMENT
  
  /// ReadToken - Used by PTHLexer to read tokens TokBuf.
  void ReadToken(Token& T);

  /// PTHMgr - The PTHManager object that created this PTHLexer.
  PTHManager& PTHMgr;
  
  Token LastFetched;
  Token EofToken;
  bool NeedsFetching;
  
public:  

  /// Create a PTHLexer for the specified token stream.
  PTHLexer(Preprocessor& pp, SourceLocation fileloc, const char* D, 
           const char* ppcond, PTHManager& PM);
  
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
    // isNextPPTokenLParen is not on the hot path, and all we care about is
    // whether or not we are at a token with kind tok::eof or tok::l_paren.
    // Just read the first byte from the current token pointer to determine
    // its kind.
    tok::TokenKind x = (tok::TokenKind) (uint8_t) *CurPtr;
    return x == tok::eof ? 2 : x == tok::l_paren;
  }    

  /// IndirectLex - An indirect call to 'Lex' that can be invoked via
  ///  the PreprocessorLexer interface.
  void IndirectLex(Token &Result) { Lex(Result); }
  
  /// getSourceLocation - Return a source location for the token in
  /// the current file.
  SourceLocation getSourceLocation() { return GetToken().getLocation(); }
  
  /// SkipBlock - Used by Preprocessor to skip the current conditional block.
  bool SkipBlock();

private:  
  /// AtLastToken - Returns true if the PTHLexer is at the last token.
  bool AtLastToken() { 
    Token T = GetToken();
    return T.is(tok::eof) ? EofToken = T, true : false;
  }
  
  /// GetToken - Returns the next token.  This method does not advance the
  ///  PTHLexer to the next token.
  Token GetToken();
  
  /// AdvanceToken - Advances the PTHLexer to the next token.
  void AdvanceToken() { NeedsFetching = true; }
  
  bool LexEndOfFile(Token &Result);
};

}  // end namespace clang

#endif
