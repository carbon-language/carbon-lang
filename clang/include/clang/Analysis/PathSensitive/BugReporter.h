// BugReporter.h - Generate PathDiagnostics  ----------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BugReporter, a utility class for generating
//  PathDiagnostics for analyses based on ValueState.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BUGREPORTER
#define LLVM_CLANG_ANALYSIS_BUGREPORTER

#include "clang/Analysis/PathSensitive/ValueState.h"
#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "llvm/ADT/SmallPtrSet.h"

// FIXME: ExplodedGraph<> should be templated on state, not the checker engine.
#include "clang/Analysis/PathSensitive/GRExprEngine.h"

namespace clang {
  
class PathDiagnostic;
class PathDiagnosticPiece;
class PathDiagnosticClient;
class ASTContext;
class Diagnostic;

class BugDescription {
public:
  BugDescription() {}
  virtual ~BugDescription() {}
  
  virtual const char* getName() const = 0;
  
  virtual const char* getDescription() const = 0;
  
  virtual PathDiagnosticPiece* getEndPath(ASTContext& Ctx,
                                          ExplodedNode<ValueState> *N) const;
  
  virtual void getRanges(const SourceRange*& beg,
                         const SourceRange*& end) const;
};
  
class BugReporterHelper {
public:
  virtual ~BugReporterHelper() {}
  
  virtual PathDiagnosticPiece* VisitNode(ExplodedNode<ValueState>* N,
                                         ExplodedNode<ValueState>* PrevN,
                                         ExplodedGraph<GRExprEngine>& G,
                                         ASTContext& Ctx) = 0;
};
  
class BugReporter {
  llvm::SmallPtrSet<void*,10> CachedErrors;
  
public:
  BugReporter() {}
  ~BugReporter();
  
  void EmitPathWarning(Diagnostic& Diag, PathDiagnosticClient* PDC,
                       ASTContext& Ctx, const BugDescription& B,
                       ExplodedGraph<GRExprEngine>& G,
                       ExplodedNode<ValueState>* N,
                       BugReporterHelper** BegHelpers = NULL,
                       BugReporterHelper** EndHelpers = NULL);
  
  void EmitWarning(Diagnostic& Diag, ASTContext& Ctx,
                   const BugDescription& B,
                   ExplodedNode<ValueState>* N);
  

private:
  bool IsCached(ExplodedNode<ValueState>* N);
  
  void GeneratePathDiagnostic(PathDiagnostic& PD, ASTContext& Ctx,
                              const BugDescription& B,
                              ExplodedGraph<GRExprEngine>& G,
                              ExplodedNode<ValueState>* N,
                              BugReporterHelper** BegHelpers = NULL,
                              BugReporterHelper** EndHelpers = NULL);
};
  
} // end clang namespace

#endif
