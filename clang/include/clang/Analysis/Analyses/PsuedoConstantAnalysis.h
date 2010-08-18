//== PsuedoConstantAnalysis.h - Find Psuedo-constants in the AST -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tracks the usage of variables in a Decl body to see if they are
// never written to, implying that they constant. This is useful in static
// analysis to see if a developer might have intended a variable to be const.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_PSUEDOCONSTANTANALYSIS
#define LLVM_CLANG_ANALYSIS_PSUEDOCONSTANTANALYSIS

#include "clang/AST/Stmt.h"

// The number of ValueDecls we want to keep track of by default (per-function)
#define VALUEDECL_SET_SIZE 256

namespace clang {

class PsuedoConstantAnalysis {
public:
  PsuedoConstantAnalysis(const Stmt *DeclBody) :
      DeclBody(DeclBody), Analyzed(false) {}
  bool isPsuedoConstant(const ValueDecl *VD);

private:
  void RunAnalysis();

  // for storing the result of analyzed ValueDecls
  llvm::SmallPtrSet<const ValueDecl*, VALUEDECL_SET_SIZE> NonConstants;

  const Stmt *DeclBody;
  bool Analyzed;
};

}

#endif
