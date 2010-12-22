//== NullDerefChecker.h - Null dereference checker --------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines NullDerefChecker and UndefDerefChecker, two builtin checks
// in ExprEngine that check for null and undefined pointers at loads
// and stores.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_DEREFCHECKER
#define LLVM_CLANG_GR_DEREFCHECKER

#include <utility>

namespace clang {

namespace GR {

class ExprEngine;
class ExplodedNode;

std::pair<ExplodedNode * const *, ExplodedNode * const *>
GetImplicitNullDereferences(ExprEngine &Eng);

} // end GR namespace

} // end clang namespace

#endif
