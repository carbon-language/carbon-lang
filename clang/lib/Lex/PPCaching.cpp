//===--- PPCaching.cpp - Handle caching lexed tokens ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements pieces of the Preprocessor interface that manage the
// caching of lexed tokens.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
using namespace clang;

void Preprocessor::CachingLex(Token &Result) {
  if (CachedLexPos < CachedTokens.size()) {
    Result = CachedTokens[CachedLexPos++];
    return;
  }

  ExitCachingLexMode();
  Lex(Result);

  if (!CacheTokens) {
    // All cached tokens were consumed.
    CachedTokens.clear();
    CachedLexPos = 0;
    return;
  }

  // We should cache the lexed token.

  EnterCachingLexMode();
  if (Result.isNot(tok::eof)) {
    CachedTokens.push_back(Result);
    ++CachedLexPos;
  }
}

void Preprocessor::EnterCachingLexMode() {
  if (InCachingLexMode())
    return;

  IncludeMacroStack.push_back(IncludeStackInfo(CurLexer, CurDirLookup,
                                               CurTokenLexer));
  CurLexer = 0;
  CurTokenLexer = 0;
}


const Token &Preprocessor::PeekAhead(unsigned N) {
  assert(CachedLexPos + N > CachedTokens.size() && "Confused caching.");
  ExitCachingLexMode();
  for (unsigned C = CachedLexPos + N - CachedTokens.size(); C > 0; --C) {
    CachedTokens.push_back(Token());
    Lex(CachedTokens.back());
  }
  EnterCachingLexMode();
  return CachedTokens.back();
}
