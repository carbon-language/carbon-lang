//===--- RewriteTest.cpp - Rewriter playground ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a testbed.
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/TokenRewriter.h"
#include "clang.h"
#include "clang/Lex/Preprocessor.h"
#include <iostream>

void clang::DoRewriteTest(Preprocessor &PP, const std::string &InFileName,
                          const std::string &OutFileName) {
  SourceManager &SM = PP.getSourceManager();
  const LangOptions &LangOpts = PP.getLangOptions();

  TokenRewriter Rewriter(SM.getMainFileID(), SM, LangOpts);
  
  
  
  
  
  std::pair<const char*,const char*> File =SM.getBufferData(SM.getMainFileID());
  
  // Create a lexer to lex all the tokens of the main file in raw mode.  Even
  // though it is in raw mode, it will not return comments.
  Lexer RawLex(SourceLocation::getFileLoc(SM.getMainFileID(), 0),
               LangOpts, File.first, File.second);
  
  RawLex.SetKeepWhitespaceMode(true);
  
  Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(tok::eof)) {
    std::cout << PP.getSpelling(RawTok);
    RawLex.LexFromRawLexer(RawTok);
  }
  
  for (TokenRewriter::token_iterator I = Rewriter.token_begin(),
       E = Rewriter.token_end(); I != E; ++I)
    std::cout << PP.getSpelling(*I);
}