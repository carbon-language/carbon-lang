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

namespace clang { namespace sema {

class AnalysisBasedWarnings {
  Sema &S;
  // The warnings to run.
  unsigned enableCheckFallThrough : 1;
  unsigned enableCheckUnreachable : 1;
  
public:

  AnalysisBasedWarnings(Sema &s);
  void IssueWarnings(const Decl *D, QualType BlockTy = QualType());
  
  void disableCheckFallThrough() { enableCheckFallThrough = 0; }
};
  
}} // end namespace clang::sema

#endif
