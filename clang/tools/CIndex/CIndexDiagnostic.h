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

struct CXUnsavedFile;

namespace llvm {
template<typename T> class SmallVectorImpl;
namespace sys { class Path; }
}

namespace clang {

class Diagnostic;
class FileManager;
class LangOptions;
class Preprocessor;
class StoredDiagnostic;
class SourceManager;

/// \brief The storage behind a CXDiagnostic
struct CXStoredDiagnostic {
  const StoredDiagnostic &Diag;
  const LangOptions &LangOpts;
  
  CXStoredDiagnostic(const StoredDiagnostic &Diag,
                     const LangOptions &LangOpts)
    : Diag(Diag), LangOpts(LangOpts) { }
};
  
/// \brief Given the path to a file that contains binary, serialized
/// diagnostics produced by Clang, load those diagnostics.
void LoadSerializedDiagnostics(const llvm::sys::Path &DiagnosticsPath,
                               unsigned num_unsaved_files,
                               struct CXUnsavedFile *unsaved_files,
                               FileManager &FileMgr,
                               SourceManager &SourceMgr,
                               llvm::SmallVectorImpl<StoredDiagnostic> &Diags);

} // end namespace clang

#endif // LLVM_CLANG_CINDEX_DIAGNOSTIC_H
