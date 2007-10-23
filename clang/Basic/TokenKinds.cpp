//===--- TokenKinds.cpp - Token Kinds Support -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the TokenKind enum and support functions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TokenKinds.h"
#include "llvm/Bitcode/Serialization.h"
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

// Serialization traits for TokenKind, PPKeywordKind, and ObjCKeywordKind

void llvm::SerializeTrait<tok::TokenKind>::Serialize(llvm::Serializer& S,
                                                     tok::TokenKind X) {
  S.EmitEnum(X,0,tok::NUM_TOKENS-1);
}

void llvm::SerializeTrait<tok::TokenKind>::Deserialize(llvm::Deserializer& D,
                                                       tok::TokenKind& X) {
  X = D.ReadEnum<tok::TokenKind>(0,tok::NUM_TOKENS-1);
}

void llvm::SerializeTrait<tok::PPKeywordKind>::Serialize(llvm::Serializer& S,
                                                       tok::PPKeywordKind X) {
  S.EmitEnum(X,0,tok::NUM_PP_KEYWORDS-1);
}

void llvm::SerializeTrait<tok::PPKeywordKind>::Deserialize(llvm::Deserializer& D,
                                                      tok::PPKeywordKind& X) {
  X = D.ReadEnum<tok::PPKeywordKind>(0,tok::NUM_PP_KEYWORDS-1);
}

void
llvm::SerializeTrait<tok::ObjCKeywordKind>::Serialize(llvm::Serializer& S,
                                                      tok::ObjCKeywordKind X) {
  S.EmitEnum(X,0,tok::NUM_OBJC_KEYWORDS-1);
}

void
llvm::SerializeTrait<tok::ObjCKeywordKind>::Deserialize(llvm::Deserializer& D,
                                                     tok::ObjCKeywordKind& X) {
  X = D.ReadEnum<tok::ObjCKeywordKind>(0,tok::NUM_OBJC_KEYWORDS-1);
}