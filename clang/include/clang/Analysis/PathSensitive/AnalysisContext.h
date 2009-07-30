//=== AnalysisContext.h - Analysis context for Path Sens analysis --*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines AnalysisContext, a class that manages the analysis context
// data for path sensitive analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSISCONTEXT_H
#define LLVM_CLANG_ANALYSIS_ANALYSISCONTEXT_H

#include "llvm/ADT/OwningPtr.h"
#include <map>

namespace clang {

class Decl;
class Stmt;
class CFG;
class LiveVariables;
class ParentMap;

/// AnalysisContext contains the context data for the function or method under
/// analysis.
class AnalysisContext {
  Decl *D;
  Stmt *Body;

  // AnalysisContext owns the following data.
  CFG *cfg;
  LiveVariables *liveness;
  ParentMap *PM;

public:
  AnalysisContext() : D(0), Body(0), cfg(0), liveness(0), PM(0) {}
  ~AnalysisContext();

  void setDecl(Decl* d) { D = d; }
  Decl *getDecl() { return D; }
  Stmt *getBody();
  CFG *getCFG();
  ParentMap &getParentMap();
  LiveVariables *getLiveVariables();
};

class AnalysisContextManager {
  std::map<Decl*, AnalysisContext> Contexts;

public:
  typedef std::map<Decl*, AnalysisContext>::iterator iterator;

  AnalysisContext *getContext(Decl *D);
};

}

#endif
