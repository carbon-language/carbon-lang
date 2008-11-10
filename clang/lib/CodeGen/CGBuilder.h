//===-- CGBuilder.h - Choose IRBuilder implementation  ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGBUILDER_H
#define CLANG_CODEGEN_CGBUILDER_H

#include "llvm/Support/IRBuilder.h"

namespace clang {
namespace CodeGen {
  // Don't preserve names on values by default.
  typedef llvm::IRBuilder<false> CGBuilderTy;
}  // end namespace CodeGen
}  // end namespace clang

#endif
