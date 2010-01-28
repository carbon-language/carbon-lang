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

#include "clang-c/Index.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"

namespace llvm { namespace sys {
class Path;
} }

namespace clang {

class Diagnostic;
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

/// \brief Given the path to a file that contains binary, serialized
/// diagnostics produced by Clang, emit those diagnostics via the
/// given diagnostic engine.
void ReportSerializedDiagnostics(const llvm::sys::Path &DiagnosticsPath,
                                 Diagnostic &Diags,
                                 unsigned num_unsaved_files,
                                 struct CXUnsavedFile *unsaved_files);

} // end namespace clang

#endif // LLVM_CLANG_CINDEX_DIAGNOSTIC_H
