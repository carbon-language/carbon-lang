/*===-- CIndexDiagnostic.h - Diagnostics C Interface ------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Implements the diagnostic functions of the Clang C interface.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
#ifndef LLVM_CLANG_CINDEX_DIAGNOSTIC_H
#define LLVM_CLANG_CINDEX_DIAGNOSTIC_H

namespace clang {

class LangOptions;
class StoredDiagnostic;

/// \brief The storage behind a CXDiagnostic
struct CXStoredDiagnostic {
  const StoredDiagnostic &Diag;
  const LangOptions &LangOpts;
  
  CXStoredDiagnostic(const StoredDiagnostic &Diag,
                     const LangOptions &LangOpts)
    : Diag(Diag), LangOpts(LangOpts) { }
};

} // end namespace clang

#endif // LLVM_CLANG_CINDEX_DIAGNOSTIC_H
