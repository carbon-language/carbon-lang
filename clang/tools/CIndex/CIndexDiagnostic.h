/*===-- CIndexDiagnostic.h - Diagnostics C Interface --------------*- C -*-===*\
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

#include "clang-c/Index.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"

namespace clang {

class Preprocessor;
  
/**
 * \brief Diagnostic client that translates Clang diagnostics into diagnostics
 * for the C interface to Clang.
 */
class CIndexDiagnosticClient : public DiagnosticClient {
  CXDiagnosticCallback Callback;
  CXClientData ClientData;
  LangOptions LangOpts;
  
public:
  CIndexDiagnosticClient(CXDiagnosticCallback Callback,
                         CXClientData ClientData)
    : Callback(Callback), ClientData(ClientData), LangOpts() { }
  
  virtual ~CIndexDiagnosticClient();
  
  virtual void BeginSourceFile(const LangOptions &LangOpts,
                               const Preprocessor *PP);
  
  virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                const DiagnosticInfo &Info);
};
  
} // end namespace clang

#endif // LLVM_CLANG_CINDEX_DIAGNOSTIC_H
