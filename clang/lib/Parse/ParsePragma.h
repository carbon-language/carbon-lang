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
  class Sema;
  class Parser;

class PragmaAlignHandler : public PragmaHandler {
  Sema &Actions;
public:
  explicit PragmaAlignHandler(Sema &A) : PragmaHandler("align"), Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaGCCVisibilityHandler : public PragmaHandler {
  Sema &Actions;
public:
  explicit PragmaGCCVisibilityHandler(Sema &A) : PragmaHandler("visibility"),
                                                 Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaOptionsHandler : public PragmaHandler {
  Sema &Actions;
public:
  explicit PragmaOptionsHandler(Sema &A) : PragmaHandler("options"),
                                           Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaPackHandler : public PragmaHandler {
  Sema &Actions;
public:
  explicit PragmaPackHandler(Sema &A) : PragmaHandler("pack"),
                                        Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaUnusedHandler : public PragmaHandler {
  Sema &Actions;
  Parser &parser;
public:
  PragmaUnusedHandler(Sema &A, Parser& p)
    : PragmaHandler("unused"), Actions(A), parser(p) {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaWeakHandler : public PragmaHandler {
  Sema &Actions;
public:
  explicit PragmaWeakHandler(Sema &A)
    : PragmaHandler("weak"), Actions(A) {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

}  // end namespace clang

#endif
