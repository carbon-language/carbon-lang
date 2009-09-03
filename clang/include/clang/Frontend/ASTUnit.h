//===--- ASTUnit.h - ASTUnit utility ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ASTUnit utility class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_ASTUNIT_H
#define LLVM_CLANG_FRONTEND_ASTUNIT_H

#include "llvm/ADT/OwningPtr.h"
#include <string>

namespace clang {
  class FileManager;
  class FileEntry;
  class SourceManager;
  class DiagnosticClient;
  class Diagnostic;
  class HeaderSearch;
  class TargetInfo;
  class Preprocessor;
  class ASTContext;
  class Decl;

/// \brief Utility class for loading a ASTContext from a PCH file.
///
class ASTUnit {
  llvm::OwningPtr<SourceManager>    SourceMgr;
  llvm::OwningPtr<DiagnosticClient> DiagClient;
  llvm::OwningPtr<Diagnostic>       Diags;
  llvm::OwningPtr<HeaderSearch>     HeaderInfo;
  llvm::OwningPtr<TargetInfo>       Target;
  llvm::OwningPtr<Preprocessor>     PP;
  llvm::OwningPtr<ASTContext>       Ctx;

  ASTUnit(const ASTUnit&); // do not implement
  ASTUnit &operator=(const ASTUnit &); // do not implement
  ASTUnit();
  
public:
  ~ASTUnit();

  const SourceManager &getSourceManager() const { return *SourceMgr.get(); }
        SourceManager &getSourceManager()       { return *SourceMgr.get(); }

  const Preprocessor &getPreprocessor() const { return *PP.get(); }
        Preprocessor &getPreprocessor()       { return *PP.get(); }
              
  const ASTContext &getASTContext() const { return *Ctx.get(); }
        ASTContext &getASTContext()       { return *Ctx.get(); }

  const Diagnostic &getDiagnostic() const { return *Diags.get(); }
        Diagnostic &getDiagnostic()       { return *Diags.get(); }

  /// \brief Create a ASTUnit from a PCH file.
  ///
  /// \param Filename PCH filename
  ///
  /// \param FileMgr The FileManager to use
  ///
  /// \param ErrMsg Error message to report if the PCH file could not be loaded
  ///
  /// \returns the initialized ASTUnit or NULL if the PCH failed to load
  static ASTUnit *LoadFromPCHFile(const std::string &Filename,
                                  FileManager &FileMgr,
                                  std::string *ErrMsg = 0);
};

} // namespace clang

#endif
