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
#include "clang/Basic/FileManager.h"
#include "clang/Index/ASTLocation.h"
#include <string>
#include <vector>
#include <cassert>
#include <utility>

namespace llvm {
  class MemoryBuffer;
}

namespace clang {
class ASTContext;
class CompilerInvocation;
class Decl;
class Diagnostic;
class FileEntry;
class FileManager;
class HeaderSearch;
class Preprocessor;
class SourceManager;
class TargetInfo;

using namespace idx;

/// \brief Utility class for loading a ASTContext from a PCH file.
///
class ASTUnit {
  FileManager FileMgr;

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

  /// Track whether the main file was loaded from an AST or not.
  bool MainFileIsAST;

  /// Track the top-level decls which appeared in an ASTUnit which was loaded
  /// from a source file.
  //
  // FIXME: This is just an optimization hack to avoid deserializing large parts
  // of a PCH file when using the Index library on an ASTUnit loaded from
  // source. In the long term we should make the Index library use efficient and
  // more scalable search mechanisms.
  std::vector<Decl*> TopLevelDecls;

  /// The name of the original source file used to generate this ASTUnit.
  std::string OriginalSourceFile;

  // Critical optimization when using clang_getCursor().
  ASTLocation LastLoc;

  ASTUnit(const ASTUnit&); // DO NOT IMPLEMENT
  ASTUnit &operator=(const ASTUnit &); // DO NOT IMPLEMENT

public:
  ASTUnit(bool MainFileIsAST);
  ~ASTUnit();

  bool isMainFileAST() const { return MainFileIsAST; }

  const SourceManager &getSourceManager() const { return SourceMgr; }
        SourceManager &getSourceManager()       { return SourceMgr; }

  const Preprocessor &getPreprocessor() const { return *PP.get(); }
        Preprocessor &getPreprocessor()       { return *PP.get(); }

  const ASTContext &getASTContext() const { return *Ctx.get(); }
        ASTContext &getASTContext()       { return *Ctx.get(); }

  const FileManager &getFileManager() const { return FileMgr; }
        FileManager &getFileManager()       { return FileMgr; }

  const std::string &getOriginalSourceFileName();
  const std::string &getPCHFileName();

  void unlinkTemporaryFile() { tempFile = true; }

  bool getOnlyLocalDecls() const { return OnlyLocalDecls; }

  void setLastASTLocation(ASTLocation ALoc) { LastLoc = ALoc; }
  ASTLocation getLastASTLocation() const { return LastLoc; }

  std::vector<Decl*> &getTopLevelDecls() {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    return TopLevelDecls;
  }
  const std::vector<Decl*> &getTopLevelDecls() const {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    return TopLevelDecls;
  }

  /// \brief A mapping from a file name to the memory buffer that stores the
  /// remapped contents of that file.
  typedef std::pair<std::string, const llvm::MemoryBuffer *> RemappedFile;
  
  /// \brief Create a ASTUnit from a PCH file.
  ///
  /// \param Filename - The PCH file to load.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  ///
  /// \returns - The initialized ASTUnit or null if the PCH failed to load.
  static ASTUnit *LoadFromPCHFile(const std::string &Filename,
                                  Diagnostic &Diags,
                                  bool OnlyLocalDecls = false,
                                  bool UseBumpAllocator = false,
                                  RemappedFile *RemappedFiles = 0,
                                  unsigned NumRemappedFiles = 0);

  /// LoadFromCompilerInvocation - Create an ASTUnit from a source file, via a
  /// CompilerInvocation object.
  ///
  /// \param CI - The compiler invocation to use; it must have exactly one input
  /// source file.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  //
  // FIXME: Move OnlyLocalDecls, UseBumpAllocator to setters on the ASTUnit, we
  // shouldn't need to specify them at construction time.
  static ASTUnit *LoadFromCompilerInvocation(const CompilerInvocation &CI,
                                             Diagnostic &Diags,
                                             bool OnlyLocalDecls = false);

  /// LoadFromCommandLine - Create an ASTUnit from a vector of command line
  /// arguments, which must specify exactly one source file.
  ///
  /// \param ArgBegin - The beginning of the argument vector.
  ///
  /// \param ArgEnd - The end of the argument vector.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  ///
  /// \param ResourceFilesPath - The path to the compiler resource files.
  //
  // FIXME: Move OnlyLocalDecls, UseBumpAllocator to setters on the ASTUnit, we
  // shouldn't need to specify them at construction time.
  static ASTUnit *LoadFromCommandLine(const char **ArgBegin,
                                      const char **ArgEnd,
                                      Diagnostic &Diags,
                                      llvm::StringRef ResourceFilesPath,
                                      bool OnlyLocalDecls = false,
                                      bool UseBumpAllocator = false,
                                      RemappedFile *RemappedFiles = 0,
                                      unsigned NumRemappedFiles = 0);
};

} // namespace clang

#endif
