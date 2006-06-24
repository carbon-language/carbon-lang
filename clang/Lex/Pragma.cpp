//===--- Pragma.cpp - Pragma registration and handling --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PragmaHandler and PragmaTable interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
using namespace llvm;
using namespace clang;

// Out-of-line destructor to provide a home for the class.
PragmaHandler::~PragmaHandler() {
}

void PragmaNamespace::HandlePragma(Preprocessor &PP, LexerToken &Tok) {
  // Read the 'namespace' that the directive is in, e.g. STDC.  Do not macro
  // expand it, the user can have a STDC #define, that should not affect this.
  PP.LexUnexpandedToken(Tok);
  
  // Get the handler for this token.  If there is no handler, ignore the pragma.
  PragmaHandler *Handler = FindHandler(Tok.getIdentifierInfo(), false);
  if (Handler == 0) return;
  
  // Otherwise, pass it down.
  Handler->HandlePragma(PP, Tok);
}
