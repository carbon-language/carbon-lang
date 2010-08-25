//=- AnalysisBasedWarnings.h - Sema warnings based on libAnalysis -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines AnalysisBasedWarnings, a worker object used by Sema
// that issues warnings based on dataflow-analysis.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_ANALYSIS_WARNINGS_H
#define LLVM_CLANG_SEMA_ANALYSIS_WARNINGS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

class BlockExpr;
class Decl;
class FunctionDecl;
class ObjCMethodDecl;
class QualType;
class Sema;

namespace sema {

class AnalysisBasedWarnings {
public:
  class Policy {
    friend class AnalysisBasedWarnings;
    // The warnings to run.
    unsigned enableCheckFallThrough : 1;
    unsigned enableCheckUnreachable : 1;
  public:
    Policy();
    void disableCheckFallThrough() { enableCheckFallThrough = 0; }
  };

private:
  Sema &S;
  Policy DefaultPolicy;

  enum VisitFlag { NotVisited = 0, Visited = 1, Pending = 2 };
  llvm::DenseMap<const FunctionDecl*, VisitFlag> VisitedFD;

  void IssueWarnings(Policy P, const Decl *D, QualType BlockTy);

public:
  AnalysisBasedWarnings(Sema &s);

  Policy getDefaultPolicy() { return DefaultPolicy; }

  void IssueWarnings(Policy P, const BlockExpr *E);
  void IssueWarnings(Policy P, const FunctionDecl *D);
  void IssueWarnings(Policy P, const ObjCMethodDecl *D);
};

}} // end namespace clang::sema

#endif
