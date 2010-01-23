//===- CIndexer.h - Clang-C Source Indexing Library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines CIndexer, a subclass of Indexer that provides extra
// functionality needed by the CIndex library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CINDEXER_H
#define LLVM_CLANG_CINDEXER_H

#include "clang-c/Index.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/ASTUnit.h"
#include "llvm/System/Path.h"
#include <vector>

using namespace clang;

/// IgnoreDiagnosticsClient - A DiagnosticsClient that just ignores emitted
/// warnings and errors.
class IgnoreDiagnosticsClient : public DiagnosticClient {
public:
  virtual ~IgnoreDiagnosticsClient() {}
  virtual void HandleDiagnostic(Diagnostic::Level, const DiagnosticInfo &) {}
};

class CIndexer {
  DiagnosticOptions DiagOpts;
  IgnoreDiagnosticsClient IgnoreDiagClient;
  llvm::OwningPtr<Diagnostic> TextDiags;
  Diagnostic IgnoreDiags;
  bool UseExternalASTGeneration;
  bool OnlyLocalDecls;
  bool DisplayDiagnostics;
  
  llvm::sys::Path ClangPath;
  
public:
  CIndexer() : IgnoreDiags(&IgnoreDiagClient), UseExternalASTGeneration(false),
               OnlyLocalDecls(false), DisplayDiagnostics(false) 
  {
    TextDiags.reset(CompilerInstance::createDiagnostics(DiagOpts, 0, 0));
  }
  
  /// \brief Whether we only want to see "local" declarations (that did not
  /// come from a previous precompiled header). If false, we want to see all
  /// declarations.
  bool getOnlyLocalDecls() const { return OnlyLocalDecls; }
  void setOnlyLocalDecls(bool Local = true) { OnlyLocalDecls = Local; }
  
  bool getDisplayDiagnostics() const { return DisplayDiagnostics; }
  void setDisplayDiagnostics(bool Display = true) {
    DisplayDiagnostics = Display;
  }
  
  bool getUseExternalASTGeneration() const { return UseExternalASTGeneration; }
  void setUseExternalASTGeneration(bool Value) {
    UseExternalASTGeneration = Value;
  }
  
  Diagnostic &getDiags() {
    return DisplayDiagnostics ? *TextDiags : IgnoreDiags;
  }
  
  /// \brief Get the path of the clang binary.
  const llvm::sys::Path& getClangPath();
  
  /// \brief Get the path of the clang resource files.
  std::string getClangResourcesPath();

  static CXString createCXString(const char *String, bool DupString = false);
};

namespace clang {
  /**
   * \brief Given a set of "unsaved" files, create temporary files and 
   * construct the clang -cc1 argument list needed to perform the remapping.
   *
   * \returns true if an error occurred.
   */
  bool RemapFiles(unsigned num_unsaved_files,
                  struct CXUnsavedFile *unsaved_files,
                  std::vector<std::string> &RemapArgs,
                  std::vector<llvm::sys::Path> &TemporaryFiles);
}

#endif
