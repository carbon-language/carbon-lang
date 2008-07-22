//==- GRAuditor.h - Observers of the creation of ExplodedNodes------*- C++ -*-//
//             
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRAuditor and its primary subclasses, an interface
//  to audit the creation of ExplodedNodes.  This interface can be used
//  to implement simple checkers that do not mutate analysis state but
//  instead operate by perfoming simple logical checks at key monitoring
//  locations (e.g., function calls).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GRAUDITOR
#define LLVM_CLANG_ANALYSIS_GRAUDITOR

#include "clang/AST/Expr.h"
#include "clang/Analysis/PathSensitive/ExplodedGraph.h"

namespace clang {
  
template <typename STATE>
class GRAuditor {
public:
  typedef ExplodedNode<STATE>       NodeTy;
  typedef typename STATE::ManagerTy ManagerTy;
  
  virtual ~GRAuditor() {}
  virtual bool Audit(NodeTy* N, ManagerTy& M) = 0;
};
  
  
} // end clang namespace

#endif
