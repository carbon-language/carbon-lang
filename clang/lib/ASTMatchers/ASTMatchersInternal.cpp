//===--- ASTMatchersInternal.cpp - Structural query framework -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Implements the base layer of the matcher framework.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"

namespace clang {
namespace ast_matchers {
namespace internal {

void BoundNodesTreeBuilder::visitMatches(Visitor *ResultVisitor) {
  if (Bindings.empty())
    Bindings.push_back(BoundNodesMap());
  for (unsigned i = 0, e = Bindings.size(); i != e; ++i) {
    ResultVisitor->visitMatch(BoundNodes(Bindings[i]));
  }
}

void BoundNodesTreeBuilder::addMatch(const BoundNodesTreeBuilder &Other) {
  for (unsigned i = 0, e = Other.Bindings.size(); i != e; ++i) {
    Bindings.push_back(Other.Bindings[i]);
  }
}

DynTypedMatcher::~DynTypedMatcher() {}

DynTypedMatcher *DynTypedMatcher::tryBind(StringRef ID) const { return NULL; }

} // end namespace internal
} // end namespace ast_matchers
} // end namespace clang
