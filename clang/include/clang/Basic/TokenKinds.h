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

namespace clang {

namespace tok {

/// TokenKind - This provides a simple uniform namespace for tokens from all C
/// languages.
enum TokenKind {
#define TOK(X) X,
#include "clang/Basic/TokenKinds.def"
  NUM_TOKENS
};

/// PPKeywordKind - This provides a namespace for preprocessor keywords which
/// start with a '#' at the beginning of the line.
enum PPKeywordKind {
#define PPKEYWORD(X) pp_##X, 
#include "clang/Basic/TokenKinds.def"
  NUM_PP_KEYWORDS
};

/// ObjCKeywordKind - This provides a namespace for Objective-C keywords which
/// start with an '@'.
enum ObjCKeywordKind {
#define OBJC1_AT_KEYWORD(X) objc_##X,
#define OBJC2_AT_KEYWORD(X) objc_##X,
#include "clang/Basic/TokenKinds.def"
  NUM_OBJC_KEYWORDS
};

const char *getTokenName(enum TokenKind Kind);

}  // end namespace tok
}  // end namespace clang

//===----------------------------------------------------------------------===//
// Serialization traits.
//===----------------------------------------------------------------------===//

namespace llvm {
  template <typename T> struct SerializeTrait;
  class Serializer;
  class Deserializer;
  
template<>
struct SerializeTrait<clang::tok::TokenKind> {
  static void Serialize(llvm::Serializer& S, clang::tok::TokenKind X);
  static void Deserialize(llvm::Deserializer& D, clang::tok::TokenKind& X);
};
  
template<>
struct SerializeTrait<clang::tok::PPKeywordKind> {
  static void Serialize(llvm::Serializer& S, clang::tok::PPKeywordKind X);
  static void Deserialize(llvm::Deserializer& D, clang::tok::PPKeywordKind& X);
};

template<>
struct SerializeTrait<clang::tok::ObjCKeywordKind> {
  static void Serialize(llvm::Serializer& S, clang::tok::ObjCKeywordKind X);
  static void Deserialize(llvm::Deserializer& D, clang::tok::ObjCKeywordKind& X);
};
  
} // end namespace llvm

#endif
