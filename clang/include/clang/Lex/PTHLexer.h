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
  
  /// Pointer to a side table containing offsets in the PTH file
  ///  for token spellings.
  const char* SpellingTable;
  
  /// Number of cached spellings left in the cached source file.
  unsigned SpellingsLeft;

  PTHLexer(const PTHLexer&);  // DO NOT IMPLEMENT
  void operator=(const PTHLexer&); // DO NOT IMPLEMENT
  
  /// ReadToken - Used by PTHLexer to read tokens TokBuf.
  void ReadToken(Token& T);

  /// PTHMgr - The PTHManager object that created this PTHLexer.
  PTHManager& PTHMgr;
  
  Token EofToken;
  
public:  

  /// Create a PTHLexer for the specified token stream.
  PTHLexer(Preprocessor& pp, SourceLocation fileloc, const char* D, 
           const char* ppcond, const char* spellingTable, unsigned numSpellings,
           PTHManager& PM);
  
  ~PTHLexer() {}
    
  /// Lex - Return the next token.
  void Lex(Token &Tok);
  
  void getEOF(Token &Tok);
  
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
    tok::TokenKind x = (tok::TokenKind) (unsigned char) *CurPtr;
    return x == tok::eof ? 2 : x == tok::l_paren;
  }    

  /// IndirectLex - An indirect call to 'Lex' that can be invoked via
  ///  the PreprocessorLexer interface.
  void IndirectLex(Token &Result) { Lex(Result); }

  /// Returns the cached spelling of a token.
  /// \param[in] sloc The SourceLocation of the token.
  /// \param[out] Buffer If a token's spelling is found in the PTH file then
  ///   upon exit from this method \c Buffer will be set to the address of
  ///   the character array representing that spelling.  No characters
  ///   are copied.
  /// \returns The number of characters for the spelling of the token.  This
  ///   value is 0 if the spelling could not be found in the PTH file.
  unsigned getSpelling(SourceLocation sloc, const char *&Buffer);
  
  /// getSourceLocation - Return a source location for the token in
  /// the current file.
  SourceLocation getSourceLocation();

  /// SkipBlock - Used by Preprocessor to skip the current conditional block.
  bool SkipBlock();
};

}  // end namespace clang

#endif
