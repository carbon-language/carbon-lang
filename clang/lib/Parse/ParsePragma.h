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

class PragmaAlignHandler : public PragmaHandler {
  Action &Actions;
public:
  explicit PragmaAlignHandler(Action &A) : PragmaHandler("align"), Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, Token &FirstToken);
};

class PragmaGCCVisibilityHandler : public PragmaHandler {
  Action &Actions;
public:
  explicit PragmaGCCVisibilityHandler(Action &A) : PragmaHandler("visibility"),
                                                   Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, Token &FirstToken);
};

class PragmaOptionsHandler : public PragmaHandler {
  Action &Actions;
public:
  explicit PragmaOptionsHandler(Action &A) : PragmaHandler("options"),
                                             Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, Token &FirstToken);
};

class PragmaPackHandler : public PragmaHandler {
  Action &Actions;
public:
  explicit PragmaPackHandler(Action &A) : PragmaHandler("pack"),
                                          Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, Token &FirstToken);
};

class PragmaUnusedHandler : public PragmaHandler {
  Action &Actions;
  Parser &parser;
public:
  PragmaUnusedHandler(Action &A, Parser& p)
    : PragmaHandler("unused"), Actions(A), parser(p) {}

  virtual void HandlePragma(Preprocessor &PP, Token &FirstToken);
};

class PragmaWeakHandler : public PragmaHandler {
  Action &Actions;
public:
  explicit PragmaWeakHandler(Action &A)
    : PragmaHandler("weak"), Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, Token &FirstToken);
};

}  // end namespace clang

#endif
