//===--- TokenKinds.h - Enum values for C Token Kinds -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TokenKind enum and support functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOKENKINDS_H
#define LLVM_CLANG_TOKENKINDS_H

namespace llvm {
namespace clang {

namespace tok {

/// TokenKind - This provides a simple uniform namespace for tokens from all C
/// languages.
enum TokenKind {
#define TOK(X) X,
#include "clang/Basic/TokenKinds.def"
  NUM_TOKENS
};

enum PPKeywordKind {
#define PPKEYWORD(X) pp_##X, 
#include "clang/Basic/TokenKinds.def"
  NUM_PP_KEYWORDS
};

const char *getTokenName(enum TokenKind Kind);

}  // end namespace tok
}  // end namespace clang
}  // end namespace llvm

#endif
