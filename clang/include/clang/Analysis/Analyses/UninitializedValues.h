//= UninitializedValues.h - Finding uses of uninitialized values --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines APIs for invoking and reported uninitialized values
// warnings.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNINIT_VALS_H
#define LLVM_CLANG_UNINIT_VALS_H

namespace clang {

class AnalysisContext;
class CFG;  
class DeclContext;
class Expr;
class VarDecl;
  
class UninitVariablesHandler {
public:
  UninitVariablesHandler() {}
  virtual ~UninitVariablesHandler();

  /// Called when the uninitialized variable is used at the given expression.
  virtual void handleUseOfUninitVariable(const Expr *ex,
                                         const VarDecl *vd,
                                         bool isAlwaysUninit) {}
  
  /// Called when the uninitialized variable analysis detects the
  /// idiom 'int x = x'.  All other uses of 'x' within the initializer
  /// are handled by handleUseOfUninitVariable.
  virtual void handleSelfInit(const VarDecl *vd) {}
};

struct UninitVariablesAnalysisStats {
  unsigned NumVariablesAnalyzed;
  unsigned NumBlockVisits;
};

void runUninitializedVariablesAnalysis(const DeclContext &dc, const CFG &cfg,
                                       AnalysisContext &ac,
                                       UninitVariablesHandler &handler,
                                       UninitVariablesAnalysisStats &stats);

}
#endif
