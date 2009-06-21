//===- AsmParser.cpp - Parser for Assembly Files --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#include "AsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool AsmParser::Run() {
  // Prime the lexer.
  Lexer.Lex();
  
  while (Lexer.isNot(asmtok::Eof))
    if (ParseStatement())
      return true;
  
  return false;
}


/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Identifier Operands* EndOfStatement
bool AsmParser::ParseStatement() {
  switch (Lexer.getKind()) {
  default:
    Lexer.PrintError(Lexer.getLoc(), "unexpected token at start of statement");
    return true;
  case asmtok::EndOfStatement:
    Lexer.Lex();
    return false;
  case asmtok::Identifier:
    break;
  // TODO: Recurse on local labels etc.
  }
  
  // If we have an identifier, handle it as the key symbol.
  //SMLoc IDLoc = Lexer.getLoc();
  std::string IDVal = Lexer.getCurStrVal();
  
  // Consume the identifier, see what is after it.
  if (Lexer.Lex() == asmtok::Colon) {
    // identifier ':'   -> Label.
    Lexer.Lex();
    return ParseStatement();
  }
  
  // Otherwise, we have a normal instruction or directive.  
  if (IDVal[0] == '.')
    outs() << "Found directive: " << IDVal << "\n";
  else
    outs() << "Found instruction: " << IDVal << "\n";

  // Skip to end of line for now.
  while (Lexer.isNot(asmtok::EndOfStatement) &&
         Lexer.isNot(asmtok::Eof))
    Lexer.Lex();
  
  // Eat EOL.
  if (Lexer.is(asmtok::EndOfStatement))
    Lexer.Lex();
  return false;
}
