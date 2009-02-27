//===--- TokenKinds.cpp - Token Kinds Support -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the TokenKind enum and support functions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TokenKinds.h"

#include <cassert>
using namespace clang;

static const char * const TokNames[] = {
#define TOK(X) #X,
#define KEYWORD(X,Y) #X,
#include "clang/Basic/TokenKinds.def"
  0
};

const char *tok::getTokenName(enum TokenKind Kind) {
  assert(Kind < tok::NUM_TOKENS);
  return TokNames[Kind];
}

const char *tok::getTokenSimpleSpelling(enum TokenKind Kind) {
  switch (Kind) {
  case tok::l_square:            return "[";
  case tok::r_square:            return "]";
  case tok::l_paren:             return "(";
  case tok::r_paren:             return ")";
  case tok::l_brace:             return "{";
  case tok::r_brace:             return "}";
  case tok::period:              return ".";
  case tok::ellipsis:            return "...";
  case tok::amp:                 return "&";
  case tok::ampamp:              return "&&";
  case tok::ampequal:            return "&=";
  case tok::star:                return "*";
  case tok::starequal:           return "*=";
  case tok::plus:                return "+";
  case tok::plusplus:            return "++";
  case tok::plusequal:           return "+=";
  case tok::minus:               return "-";
  case tok::arrow:               return "->";
  case tok::minusminus:          return "--";
  case tok::minusequal:          return "-=";
  case tok::tilde:               return "~";
  case tok::exclaim:             return "!";
  case tok::exclaimequal:        return "!=";
  case tok::slash:               return "/";
  case tok::slashequal:          return "/=";
  case tok::percent:             return "%";
  case tok::percentequal:        return "%=";
  case tok::less:                return "<";
  case tok::lessless:            return "<<";
  case tok::lessequal:           return "<=";
  case tok::lesslessequal:       return "<<=";
  case tok::greater:             return ">";
  case tok::greatergreater:      return ">>";
  case tok::greaterequal:        return ">=";
  case tok::greatergreaterequal: return ">>=";
  case tok::caret:               return "^";
  case tok::caretequal:          return "^=";
  case tok::pipe:                return "|";
  case tok::pipepipe:            return "||";
  case tok::pipeequal:           return "|=";
  case tok::question:            return "?";
  case tok::colon:               return ":";
  case tok::semi:                return ";";
  case tok::equal:               return "=";
  case tok::equalequal:          return "==";
  case tok::comma:               return ",";
  case tok::hash:                return "#";
  case tok::hashhash:            return "##";
  case tok::hashat:              return "#@";
  case tok::periodstar:          return ".*";
  case tok::arrowstar:           return "->*";
  case tok::coloncolon:          return "::";
  case tok::at:                  return "@";
  default: break;
  }

  return 0;
}
