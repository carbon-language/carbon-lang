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

#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/OwningPtr.h"
#include <string>

namespace clang {
  class FileManager;
  class FileEntry;
  class SourceManager;
  class Diagnostic;
  class HeaderSearch;
  class TargetInfo;
  class Preprocessor;
  class ASTContext;
  class Decl;

/// \brief Utility class for loading a ASTContext from a PCH file.
///
class ASTUnit {
  Diagnostic                       &Diags;
  SourceManager                     SourceMgr;
  llvm::OwningPtr<HeaderSearch>     HeaderInfo;
  llvm::OwningPtr<TargetInfo>       Target;
  llvm::OwningPtr<Preprocessor>     PP;
  llvm::OwningPtr<ASTContext>       Ctx;
  bool                              tempFile;
  
  // OnlyLocalDecls - when true, walking this AST should only visit declarations
  // that come from the AST itself, not from included precompiled headers.
  // FIXME: This is temporary; eventually, CIndex will always do this.
  bool                              OnlyLocalDecls;
  
  ASTUnit(const ASTUnit&); // DO NOT IMPLEMENT
  ASTUnit &operator=(const ASTUnit &); // DO NOT IMPLEMENT
  ASTUnit(Diagnostic &_Diag);

public:
  ~ASTUnit();

  const SourceManager &getSourceManager() const { return SourceMgr; }
        SourceManager &getSourceManager()       { return SourceMgr; }

  const Preprocessor &getPreprocessor() const { return *PP.get(); }
        Preprocessor &getPreprocessor()       { return *PP.get(); }

  const ASTContext &getASTContext() const { return *Ctx.get(); }
        ASTContext &getASTContext()       { return *Ctx.get(); }

  const Diagnostic &getDiagnostic() const { return Diags; }
        Diagnostic &getDiagnostic()       { return Diags; }

  FileManager &getFileManager();
  const std::string &getOriginalSourceFileName();
  const std::string &getPCHFileName();

  void unlinkTemporaryFile() { tempFile = true; }
  
  bool getOnlyLocalDecls() const { return OnlyLocalDecls; }
  
  /// \brief Create a ASTUnit from a PCH file.
  ///
  /// \param Filename - The PCH file to load.
  ///
  /// \param Diags - The Diagnostic implementation to use.
  ///
  /// \param FileMgr - The FileManager to use.
  ///
  /// \param ErrMsg - Error message to report if the PCH file could not be
  /// loaded.
  ///
  /// \returns - The initialized ASTUnit or null if the PCH failed to load.
  static ASTUnit *LoadFromPCHFile(const std::string &Filename,
                                  Diagnostic &Diags,
                                  FileManager &FileMgr,
                                  std::string *ErrMsg = 0,
                                  bool OnlyLocalDecls = false);
};

} // namespace clang

#endif
