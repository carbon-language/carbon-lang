//===--- AccessSpecifier.h - C++ Access Specifiers -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces used for C++ access specifiers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_ACCESS_SPECIFIER_H
#define LLVM_CLANG_PARSE_ACCESS_SPECIFIER_H

namespace clang {

/// AccessSpecifier - A C++ access specifier (none, public, private,
/// protected).
enum AccessSpecifier {
  AS_none,
  AS_public,
  AS_protected,
  AS_private
};

} // end namespace clang

#endif
