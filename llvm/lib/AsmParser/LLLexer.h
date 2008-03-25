//===- LLLexer.h - Lexer for LLVM Assembly Files ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the Lexer for .ll files.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_ASMPARSER_LLLEXER_H
#define LIB_ASMPARSER_LLLEXER_H

#include <vector>
#include <string>
#include <iosfwd>

namespace llvm {
  class MemoryBuffer;
  
  class LLLexer {
    const char *CurPtr;
    unsigned CurLineNo;
    MemoryBuffer *CurBuf;
    
    const char *TokStart;
    
    std::string TheError;
  public:
    explicit LLLexer(MemoryBuffer *StartBuf);
    ~LLLexer() {}

    const char *getTokStart() const { return TokStart; }
    unsigned getTokLength() const { return CurPtr-TokStart; }
    unsigned getLineNo() const { return CurLineNo; }
    std::string getFilename() const;
    int LexToken();
    
    const std::string getError() const { return TheError; }
    
  private:
    int getNextChar();
    void SkipLineComment();
    int LexIdentifier();
    int LexDigitOrNegative();
    int LexPositive();
    int LexAt();
    int LexPercent();
    int LexQuote();
    int Lex0x();
  };
} // end namespace llvm

#endif
