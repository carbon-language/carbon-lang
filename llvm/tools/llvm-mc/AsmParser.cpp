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
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool AsmParser::Error(SMLoc L, const char *Msg) {
  Lexer.PrintMessage(L, Msg);
  return true;
}

bool AsmParser::TokError(const char *Msg) {
  Lexer.PrintMessage(Lexer.getLoc(), Msg);
  return true;
}

bool AsmParser::Run() {
  // Prime the lexer.
  Lexer.Lex();
  
  while (Lexer.isNot(asmtok::Eof))
    if (ParseStatement())
      return true;
  
  return false;
}

/// EatToEndOfStatement - Throw away the rest of the line for testing purposes.
void AsmParser::EatToEndOfStatement() {
  while (Lexer.isNot(asmtok::EndOfStatement) &&
         Lexer.isNot(asmtok::Eof))
    Lexer.Lex();
  
  // Eat EOL.
  if (Lexer.is(asmtok::EndOfStatement))
    Lexer.Lex();
}


/// ParseParenExpr - Parse a paren expression and return it.
/// NOTE: This assumes the leading '(' has already been consumed.
///
/// parenexpr ::= expr)
///
bool AsmParser::ParseParenExpr(int64_t &Res) {
  if (ParseExpression(Res)) return true;
  if (Lexer.isNot(asmtok::RParen))
    return TokError("expected ')' in parentheses expression");
  Lexer.Lex();
  return false;
}

/// ParsePrimaryExpr - Parse a primary expression and return it.
///  primaryexpr ::= (parenexpr
///  primaryexpr ::= symbol
///  primaryexpr ::= number
///  primaryexpr ::= ~,+,- primaryexpr
bool AsmParser::ParsePrimaryExpr(int64_t &Res) {
  switch (Lexer.getKind()) {
  default:
    return TokError("unknown token in expression");
  case asmtok::Identifier:
    // This is a label, this should be parsed as part of an expression, to
    // handle things like LFOO+4
    Res = 0; // FIXME.
    Lexer.Lex(); // Eat identifier.
    return false;
  case asmtok::IntVal:
    Res = Lexer.getCurIntVal();
    Lexer.Lex(); // Eat identifier.
    return false;
  case asmtok::LParen:
    Lexer.Lex(); // Eat the '('.
    return ParseParenExpr(Res);
  case asmtok::Tilde:
  case asmtok::Plus:
  case asmtok::Minus:
    Lexer.Lex(); // Eat the operator.
    return ParsePrimaryExpr(Res);
  }
}

/// ParseExpression - Parse an expression and return it.
/// 
///  expr ::= expr +,- expr          -> lowest.
///  expr ::= expr |,^,&,! expr      -> middle.
///  expr ::= expr *,/,%,<<,>> expr  -> highest.
///  expr ::= primaryexpr
///
bool AsmParser::ParseExpression(int64_t &Res) {
  return ParsePrimaryExpr(Res) ||
         ParseBinOpRHS(1, Res);
}

static unsigned getBinOpPrecedence(asmtok::TokKind K) {
  switch (K) {
  default: return 0;    // not a binop.
  case asmtok::Plus:
  case asmtok::Minus:
    return 1;
  case asmtok::Pipe:
  case asmtok::Caret:
  case asmtok::Amp:
  case asmtok::Exclaim:
    return 2;
  case asmtok::Star:
  case asmtok::Slash:
  case asmtok::Percent:
  case asmtok::LessLess:
  case asmtok::GreaterGreater:
    return 3;
  }
}


/// ParseBinOpRHS - Parse all binary operators with precedence >= 'Precedence'.
/// Res contains the LHS of the expression on input.
bool AsmParser::ParseBinOpRHS(unsigned Precedence, int64_t &Res) {
  while (1) {
    unsigned TokPrec = getBinOpPrecedence(Lexer.getKind());
    
    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.
    if (TokPrec < Precedence)
      return false;
    
    //asmtok::TokKind BinOp = Lexer.getKind();
    Lexer.Lex();
    
    // Eat the next primary expression.
    int64_t RHS;
    if (ParsePrimaryExpr(RHS)) return true;
    
    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    unsigned NextTokPrec = getBinOpPrecedence(Lexer.getKind());
    if (TokPrec < NextTokPrec) {
      if (ParseBinOpRHS(Precedence+1, RHS)) return true;
    }

    // Merge LHS/RHS: fixme use the right operator etc.
    Res += RHS;
  }
}

  
  
  
/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Directive ...Operands... EndOfStatement
///   ::= Label* Identifier OperandList* EndOfStatement
bool AsmParser::ParseStatement() {
  switch (Lexer.getKind()) {
  default:
    return TokError("unexpected token at start of statement");
  case asmtok::EndOfStatement:
    Lexer.Lex();
    return false;
  case asmtok::Identifier:
    break;
  // TODO: Recurse on local labels etc.
  }
  
  // If we have an identifier, handle it as the key symbol.
  SMLoc IDLoc = Lexer.getLoc();
  const char *IDVal = Lexer.getCurStrVal();
  
  // Consume the identifier, see what is after it.
  if (Lexer.Lex() == asmtok::Colon) {
    // identifier ':'   -> Label.
    Lexer.Lex();
    
    // Since we saw a label, create a symbol and emit it.
    // FIXME: If the label starts with L it is an assembler temporary label.
    // Why does the client of this api need to know this?
    Out.EmitLabel(Ctx.GetOrCreateSymbol(IDVal));
    
    return ParseStatement();
  }
  
  // Otherwise, we have a normal instruction or directive.  
  if (IDVal[0] == '.') {
    if (!strcmp(IDVal, ".section"))
      return ParseDirectiveSection();
    
    
    Lexer.PrintMessage(IDLoc, "warning: ignoring directive for now");
    EatToEndOfStatement();
    return false;
  }


  MCInst Inst;
  if (ParseX86InstOperands(Inst))
    return true;
  
  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in argument list");

  // Eat the end of statement marker.
  Lexer.Lex();
  
  // Instruction is good, process it.
  outs() << "Found instruction: " << IDVal << " with " << Inst.getNumOperands()
         << " operands.\n";
  
  // Skip to end of line for now.
  return false;
}

/// ParseDirectiveSection:
///   ::= .section identifier
bool AsmParser::ParseDirectiveSection() {
  if (Lexer.isNot(asmtok::Identifier))
    return TokError("expected identifier after '.section' directive");
  
  std::string Section = Lexer.getCurStrVal();
  Lexer.Lex();
  
  // Accept a comma separated list of modifiers.
  while (Lexer.is(asmtok::Comma)) {
    Lexer.Lex();
    
    if (Lexer.isNot(asmtok::Identifier))
      return TokError("expected identifier in '.section' directive");
    Section += ',';
    Section += Lexer.getCurStrVal();
    Lexer.Lex();
  }
  
  if (Lexer.isNot(asmtok::EndOfStatement))
    return TokError("unexpected token in '.section' directive");
  Lexer.Lex();

  Out.SwitchSection(Ctx.GetSection(Section.c_str()));
  return false;
}

