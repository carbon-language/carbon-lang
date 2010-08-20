//===- ASTCommon.h - Common stuff for ASTReader/ASTWriter -*- C++ -*-=========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines common functions that both ASTReader and ASTWriter use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_LIB_AST_COMMON_H
#define LLVM_CLANG_SERIALIZATION_LIB_AST_COMMON_H

namespace clang {
  class Selector;

namespace serialization {

unsigned ComputeHash(Selector Sel);

} // namespace serialization

} // namespace clang

#endif
