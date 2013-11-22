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

DynTypedMatcher::MatcherStorage::~MatcherStorage() {}

void BoundNodesTreeBuilder::addMatch(const BoundNodesTreeBuilder &Other) {
  for (unsigned i = 0, e = Other.Bindings.size(); i != e; ++i) {
    Bindings.push_back(Other.Bindings[i]);
  }
}

bool NotUnaryOperator(const ast_type_traits::DynTypedNode DynNode,
                      ASTMatchFinder *Finder, BoundNodesTreeBuilder *Builder,
                      ArrayRef<DynTypedMatcher> InnerMatchers) {
  if (InnerMatchers.size() != 1)
    return false;

  // The 'unless' matcher will always discard the result:
  // If the inner matcher doesn't match, unless returns true,
  // but the inner matcher cannot have bound anything.
  // If the inner matcher matches, the result is false, and
  // any possible binding will be discarded.
  // We still need to hand in all the bound nodes up to this
  // point so the inner matcher can depend on bound nodes,
  // and we need to actively discard the bound nodes, otherwise
  // the inner matcher will reset the bound nodes if it doesn't
  // match, but this would be inversed by 'unless'.
  BoundNodesTreeBuilder Discard(*Builder);
  return !InnerMatchers[0].matches(DynNode, Finder, &Discard);
}

bool AllOfVariadicOperator(const ast_type_traits::DynTypedNode DynNode,
                           ASTMatchFinder *Finder,
                           BoundNodesTreeBuilder *Builder,
                           ArrayRef<DynTypedMatcher> InnerMatchers) {
  // allOf leads to one matcher for each alternative in the first
  // matcher combined with each alternative in the second matcher.
  // Thus, we can reuse the same Builder.
  for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
    if (!InnerMatchers[i].matches(DynNode, Finder, Builder))
      return false;
  }
  return true;
}

bool EachOfVariadicOperator(const ast_type_traits::DynTypedNode DynNode,
                            ASTMatchFinder *Finder,
                            BoundNodesTreeBuilder *Builder,
                            ArrayRef<DynTypedMatcher> InnerMatchers) {
  BoundNodesTreeBuilder Result;
  bool Matched = false;
  for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
    BoundNodesTreeBuilder BuilderInner(*Builder);
    if (InnerMatchers[i].matches(DynNode, Finder, &BuilderInner)) {
      Matched = true;
      Result.addMatch(BuilderInner);
    }
  }
  *Builder = Result;
  return Matched;
}

bool AnyOfVariadicOperator(const ast_type_traits::DynTypedNode DynNode,
                           ASTMatchFinder *Finder,
                           BoundNodesTreeBuilder *Builder,
                           ArrayRef<DynTypedMatcher> InnerMatchers) {
  for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
    BoundNodesTreeBuilder Result = *Builder;
    if (InnerMatchers[i].matches(DynNode, Finder, &Result)) {
      *Builder = Result;
      return true;
    }
  }
  return false;
}

} // end namespace internal
} // end namespace ast_matchers
} // end namespace clang
