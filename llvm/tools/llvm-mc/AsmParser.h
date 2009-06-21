//===- AsmParser.h - Parser for Assembly Files ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class declares the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef ASMPARSER_H
#define ASMPARSER_H

#include "AsmLexer.h"

namespace llvm {

class AsmParser {
  AsmLexer Lexer;
  
public:
  AsmParser(SourceMgr &SM) : Lexer(SM) {}
  ~AsmParser() {}
  
  bool Run();
  
private:
  bool ParseStatement();
  
};

} // end namespace llvm

#endif
