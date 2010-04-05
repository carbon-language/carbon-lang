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

#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/OwningPtr.h"
#include "clang/Basic/FileManager.h"
#include "clang/Index/ASTLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/System/Path.h"
#include <map>
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
public:
  typedef std::map<FileID, std::vector<PreprocessedEntity *> > 
    PreprocessedEntitiesByFileMap;
private:
  llvm::IntrusiveRefCntPtr<Diagnostic> Diagnostics;
  llvm::OwningPtr<FileManager>      FileMgr;
  llvm::OwningPtr<SourceManager>    SourceMgr;
  llvm::OwningPtr<HeaderSearch>     HeaderInfo;
  llvm::OwningPtr<TargetInfo>       Target;
  llvm::OwningPtr<Preprocessor>     PP;
  llvm::OwningPtr<ASTContext>       Ctx;
  
  /// Optional owned invocation, just used to make the invocation used in
  /// LoadFromCommandLine available.
  llvm::OwningPtr<CompilerInvocation> Invocation;

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

  /// \brief The set of diagnostics produced when creating this
  /// translation unit.
  llvm::SmallVector<StoredDiagnostic, 4> StoredDiagnostics;

  /// \brief Temporary files that should be removed when the ASTUnit is 
  /// destroyed.
  llvm::SmallVector<llvm::sys::Path, 4> TemporaryFiles;

  /// \brief A mapping from file IDs to the set of preprocessed entities
  /// stored in that file. 
  ///
  /// FIXME: This is just an optimization hack to avoid searching through
  /// many preprocessed entities during cursor traversal in the CIndex library.
  /// Ideally, we would just be able to perform a binary search within the
  /// list of preprocessed entities.
  PreprocessedEntitiesByFileMap PreprocessedEntitiesByFile;
  
  /// \brief Simple hack to allow us to assert that ASTUnit is not being
  /// used concurrently, which is not supported.
  ///
  /// Clients should create instances of the ConcurrencyCheck class whenever
  /// using the ASTUnit in a way that isn't intended to be concurrent, which is
  /// just about any usage.
  unsigned int ConcurrencyCheckValue;
  static const unsigned int CheckLocked = 28573289;
  static const unsigned int CheckUnlocked = 9803453;
  
  ASTUnit(const ASTUnit&); // DO NOT IMPLEMENT
  ASTUnit &operator=(const ASTUnit &); // DO NOT IMPLEMENT
  
  explicit ASTUnit(bool MainFileIsAST);

public:
  class ConcurrencyCheck {
    volatile ASTUnit &Self;
    
  public:
    explicit ConcurrencyCheck(ASTUnit &Self)
      : Self(Self) 
    { 
      assert(Self.ConcurrencyCheckValue == CheckUnlocked && 
             "Concurrent access to ASTUnit!");
      Self.ConcurrencyCheckValue = CheckLocked;
    }
    
    ~ConcurrencyCheck() {
      Self.ConcurrencyCheckValue = CheckUnlocked;
    }
  };
  friend class ConcurrencyCheck;
  
  ~ASTUnit();

  bool isMainFileAST() const { return MainFileIsAST; }

  const Diagnostic &getDiagnostics() const { return *Diagnostics; }
  Diagnostic &getDiagnostics()             { return *Diagnostics; }
  
  const SourceManager &getSourceManager() const { return *SourceMgr; }
        SourceManager &getSourceManager()       { return *SourceMgr; }

  const Preprocessor &getPreprocessor() const { return *PP.get(); }
        Preprocessor &getPreprocessor()       { return *PP.get(); }

  const ASTContext &getASTContext() const { return *Ctx.get(); }
        ASTContext &getASTContext()       { return *Ctx.get(); }

  const FileManager &getFileManager() const { return *FileMgr; }
        FileManager &getFileManager()       { return *FileMgr; }

  const std::string &getOriginalSourceFileName();
  const std::string &getPCHFileName();

  /// \brief Add a temporary file that the ASTUnit depends on.
  ///
  /// This file will be erased when the ASTUnit is destroyed.
  void addTemporaryFile(const llvm::sys::Path &TempFile) {
    TemporaryFiles.push_back(TempFile);
  }
                        
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

  /// \brief Retrieve the mapping from File IDs to the preprocessed entities
  /// within that file.
  PreprocessedEntitiesByFileMap &getPreprocessedEntitiesByFile() {
    return PreprocessedEntitiesByFile;
  }
  
  // Retrieve the diagnostics associated with this AST
  typedef const StoredDiagnostic *stored_diag_iterator;
  stored_diag_iterator stored_diag_begin() const { 
    return StoredDiagnostics.begin(); 
  }
  stored_diag_iterator stored_diag_end() const { 
    return StoredDiagnostics.end(); 
  }
  unsigned stored_diag_size() const { return StoredDiagnostics.size(); }
  
  llvm::SmallVector<StoredDiagnostic, 4> &getStoredDiagnostics() { 
    return StoredDiagnostics; 
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
                                  llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                  bool OnlyLocalDecls = false,
                                  RemappedFile *RemappedFiles = 0,
                                  unsigned NumRemappedFiles = 0,
                                  bool CaptureDiagnostics = false);

  /// LoadFromCompilerInvocation - Create an ASTUnit from a source file, via a
  /// CompilerInvocation object.
  ///
  /// \param CI - The compiler invocation to use; it must have exactly one input
  /// source file. The ASTUnit takes ownership of the CompilerInvocation object.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  //
  // FIXME: Move OnlyLocalDecls, UseBumpAllocator to setters on the ASTUnit, we
  // shouldn't need to specify them at construction time.
  static ASTUnit *LoadFromCompilerInvocation(CompilerInvocation *CI,
                                     llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                             bool OnlyLocalDecls = false,
                                             bool CaptureDiagnostics = false);

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
                                    llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                      llvm::StringRef ResourceFilesPath,
                                      bool OnlyLocalDecls = false,
                                      RemappedFile *RemappedFiles = 0,
                                      unsigned NumRemappedFiles = 0,
                                      bool CaptureDiagnostics = false);
};

} // namespace clang

#endif
