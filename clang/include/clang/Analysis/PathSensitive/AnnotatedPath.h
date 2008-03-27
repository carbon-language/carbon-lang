//=-- AnnotatedPath.h - An annotated list of ExplodedNodes -*- C++ -*-------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines AnnotatedPath, which represents a collection of
//  annotated ExplodedNodes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANNOTPATH
#define LLVM_CLANG_ANALYSIS_ANNOTPATH

#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include <string>
#include <list>

namespace clang {

  class Expr;

template <typename STATE>
class AnnotatedNode {
  ExplodedNode<STATE> *Node;
  std::string annotation;
  Expr* E;

public:
  AnnotatedNode(ExplodedNode<STATE>* N, const std::string& annot,
                Expr* e = NULL)
  : Node(N), annotation(annot), E(e) {}

  ExplodedNode<STATE>* getNode() const { return Node; }
  
  const std::string& getString() const { return annotation; }
  
  Expr* getExpr() const { return E; }
};
  

template <typename STATE>
class AnnotatedPath {
  typedef std::list<AnnotatedNode<STATE> >  impl;
  impl path;
public:
  AnnotatedPath() {}
  
  void push_back(ExplodedNode<STATE>* N, const std::string& s, Expr* E = NULL) {
    path.push_back(AnnotatedNode<STATE>(N, s, E));
  }
  
  typedef typename impl::iterator iterator;
  
  iterator begin() { return path.begin(); }
  iterator end() { return path.end(); }
  
  AnnotatedNode<STATE>& back() { return path.back(); }
  const AnnotatedNode<STATE>& back() const { return path.back(); }
};
  
} // end clang namespace

#endif
