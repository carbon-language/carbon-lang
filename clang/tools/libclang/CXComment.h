//===- CXComment.h - Routines for manipulating CXComments -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines routines for manipulating CXComments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CXCOMMENT_H
#define LLVM_CLANG_CXCOMMENT_H

#include "clang-c/Index.h"

#include "clang/AST/Comment.h"

namespace clang {
namespace cxcomment {

inline CXComment createCXComment(const comments::Comment *C) {
  CXComment Result;
  Result.Data = C;
  return Result;
}

inline const comments::Comment *getASTNode(CXComment CXC) {
  return static_cast<const comments::Comment *>(CXC.Data);
}

template<typename T>
inline const T *getASTNodeAs(CXComment CXC) {
  const comments::Comment *C = getASTNode(CXC);
  if (!C)
    return NULL;

  return dyn_cast<T>(C);
}

} // end namespace cxcomment
} // end namespace clang

#endif

