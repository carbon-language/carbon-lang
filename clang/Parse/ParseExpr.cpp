//===--- Expression.cpp - Expression Parsing ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Expression parsing implementation.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
using namespace llvm;
using namespace clang;

Parser::ExprTy Parser::ParseExpression() {
  if (Tok.getKind() == tok::numeric_constant) {
    ConsumeToken();
    return 0;
  }
  
  Diag(Tok, diag::err_parse_error);
  return 0;
}

///primary-expression:
///  identifier
///  constant
///  string-literal
///  '(' expression ')'

/// ParseParenExpression - C99 c.5.1p5
///       primary-expression:
///         '(' expression ')'
void Parser::ParseParenExpression() {
  assert(Tok.getKind() == tok::l_paren && "Not a paren expr!");
  SourceLocation OpenLoc = Tok.getLocation();
  ConsumeParen();
  
  ParseExpression();
  
  if (Tok.getKind() == tok::r_paren) {
    ConsumeParen();
  } else {
    Diag(Tok, diag::err_expected_rparen);
    Diag(OpenLoc, diag::err_matching);
  }
}
