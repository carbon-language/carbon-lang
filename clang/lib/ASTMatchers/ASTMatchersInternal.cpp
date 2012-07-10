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

BoundNodesTree::BoundNodesTree() {}

BoundNodesTree::BoundNodesTree(
  const std::map<std::string, const Decl*>& DeclBindings,
  const std::map<std::string, const Stmt*>& StmtBindings,
  const std::vector<BoundNodesTree> RecursiveBindings)
  : DeclBindings(DeclBindings), StmtBindings(StmtBindings),
    RecursiveBindings(RecursiveBindings) {}

void BoundNodesTree::copyTo(BoundNodesTreeBuilder* Builder) const {
  copyBindingsTo(DeclBindings, Builder);
  copyBindingsTo(StmtBindings, Builder);
  for (std::vector<BoundNodesTree>::const_iterator
         I = RecursiveBindings.begin(),
         E = RecursiveBindings.end();
       I != E; ++I) {
    Builder->addMatch(*I);
  }
}

template <typename T>
void BoundNodesTree::copyBindingsTo(
    const T& Bindings, BoundNodesTreeBuilder* Builder) const {
  for (typename T::const_iterator I = Bindings.begin(),
                                  E = Bindings.end();
       I != E; ++I) {
    Builder->setBinding(I->first, I->second);
  }
}

void BoundNodesTree::visitMatches(Visitor* ResultVisitor) {
  std::map<std::string, const Decl*> AggregatedDeclBindings;
  std::map<std::string, const Stmt*> AggregatedStmtBindings;
  visitMatchesRecursively(ResultVisitor, AggregatedDeclBindings,
                          AggregatedStmtBindings);
}

void BoundNodesTree::
visitMatchesRecursively(Visitor* ResultVisitor,
                        std::map<std::string, const Decl*>
                          AggregatedDeclBindings,
                        std::map<std::string, const Stmt*>
                          AggregatedStmtBindings) {
  copy(DeclBindings.begin(), DeclBindings.end(),
       inserter(AggregatedDeclBindings, AggregatedDeclBindings.begin()));
  copy(StmtBindings.begin(), StmtBindings.end(),
       inserter(AggregatedStmtBindings, AggregatedStmtBindings.begin()));
  if (RecursiveBindings.empty()) {
    ResultVisitor->visitMatch(BoundNodes(AggregatedDeclBindings,
                                         AggregatedStmtBindings));
  } else {
    for (unsigned I = 0; I < RecursiveBindings.size(); ++I) {
      RecursiveBindings[I].visitMatchesRecursively(ResultVisitor,
                                                   AggregatedDeclBindings,
                                                   AggregatedStmtBindings);
    }
  }
}

BoundNodesTreeBuilder::BoundNodesTreeBuilder() {}

void BoundNodesTreeBuilder::setBinding(const std::string &Id,
                                       const Decl *Node) {
  DeclBindings[Id] = Node;
}

void BoundNodesTreeBuilder::setBinding(const std::string &Id,
                                       const Stmt *Node) {
  StmtBindings[Id] = Node;
}

void BoundNodesTreeBuilder::addMatch(const BoundNodesTree& Bindings) {
  RecursiveBindings.push_back(Bindings);
}

BoundNodesTree BoundNodesTreeBuilder::build() const {
  return BoundNodesTree(DeclBindings, StmtBindings, RecursiveBindings);
}

} // end namespace internal
} // end namespace ast_matchers
} // end namespace clang
