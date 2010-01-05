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
#include "clang/Index/ASTLocation.h"
#include "clang/Index/Indexer.h"
#include "clang/Index/Program.h"
#include "clang/Index/Utils.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/ASTUnit.h"
#include "llvm/System/Path.h"

using namespace clang;

/// IgnoreDiagnosticsClient - A DiagnosticsClient that just ignores emitted
/// warnings and errors.
class IgnoreDiagnosticsClient : public DiagnosticClient {
public:
  virtual ~IgnoreDiagnosticsClient() {}
  virtual void HandleDiagnostic(Diagnostic::Level, const DiagnosticInfo &) {}
};

class CIndexer : public Indexer {
  DiagnosticOptions DiagOpts;
  IgnoreDiagnosticsClient IgnoreDiagClient;
  llvm::OwningPtr<Diagnostic> TextDiags;
  Diagnostic IgnoreDiags;
  bool UseExternalASTGeneration;
  bool OnlyLocalDecls;
  bool DisplayDiagnostics;
  
  llvm::sys::Path ClangPath;
  
public:
  explicit CIndexer(Program *prog) : Indexer(*prog),
    IgnoreDiags(&IgnoreDiagClient),
    UseExternalASTGeneration(false),
    OnlyLocalDecls(false),
    DisplayDiagnostics(false) {
    TextDiags.reset(CompilerInstance::createDiagnostics(DiagOpts, 0, 0));
  }
  
  virtual ~CIndexer() { delete &getProgram(); }
  
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
};

#endif
