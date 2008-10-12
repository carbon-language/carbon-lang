//===--- TokenRewriter.cpp - Token-based code rewriting interface ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the TokenRewriter class, which is used for code
//  transformations.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/TokenRewriter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;

TokenRewriter::TokenRewriter(unsigned FileID, SourceManager &SM,
                             const LangOptions &LangOpts) {
  
  std::pair<const char*,const char*> File = SM.getBufferData(FileID);
  
  // Create a lexer to lex all the tokens of the main file in raw mode.
  Lexer RawLex(SourceLocation::getFileLoc(FileID, 0),
               LangOpts, File.first, File.second);
  
  // Return all comments and whitespace as tokens.
  RawLex.SetKeepWhitespaceMode(true);

  // Lex the file, populating our datastructures.
  Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(tok::eof)) {
    AddToken(RawTok, TokenList.end());
    RawLex.LexFromRawLexer(RawTok);
  }
  
  
}

/// AddToken - Add the specified token into the Rewriter before the other
/// position.
void TokenRewriter::AddToken(const Token &T, TokenRefTy Where) {
  Where = TokenList.insert(Where, T);
  
  bool InsertSuccess = TokenAtLoc.insert(std::make_pair(T.getLocation(),
                                                        Where)).second;
  assert(InsertSuccess && "Token location already in rewriter!");
  InsertSuccess = InsertSuccess;
}
    
