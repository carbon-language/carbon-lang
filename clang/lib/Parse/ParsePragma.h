//===---- ParserPragmas.h - Language specific pragmas -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines #pragma handlers for language specific pragmas.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSEPRAGMA_H
#define LLVM_CLANG_PARSE_PARSEPRAGMA_H

#include "clang/Lex/Pragma.h"

namespace clang {
  class Action;
  class Parser;

class PragmaPackHandler : public PragmaHandler {
  Action &Actions;
public:
  PragmaPackHandler(const IdentifierInfo *N, Action &A) : PragmaHandler(N), 
                                                          Actions(A) {}
  
  virtual void HandlePragma(Preprocessor &PP, Token &FirstToken);  
};
  
class PragmaUnusedHandler : public PragmaHandler {
  Action &Actions;
  Parser &parser;
public:
  PragmaUnusedHandler(const IdentifierInfo *N, Action &A, Parser& p)
    : PragmaHandler(N), Actions(A), parser(p) {}
  
  virtual void HandlePragma(Preprocessor &PP, Token &FirstToken);  
};  

}  // end namespace clang

#endif
