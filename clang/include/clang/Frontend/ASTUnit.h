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

#include "clang/Index/ASTLocation.h"
#include "clang/Serialization/PCHBitCodes.h"
#include "clang/Sema/Sema.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang-c/Index.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/System/Path.h"
#include "llvm/Support/Timer.h"
#include <map>
#include <string>
#include <vector>
#include <cassert>
#include <utility>
#include <sys/types.h>

namespace llvm {
  class MemoryBuffer;
}

namespace clang {
class ASTContext;
class CodeCompleteConsumer;
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
  
  /// \brief The AST consumer that received information about the translation
  /// unit as it was parsed or loaded.
  llvm::OwningPtr<ASTConsumer> Consumer;
  
  /// \brief The semantic analysis object used to type-check the translation
  /// unit.
  llvm::OwningPtr<Sema> TheSema;
  
  /// Optional owned invocation, just used to make the invocation used in
  /// LoadFromCommandLine available.
  llvm::OwningPtr<CompilerInvocation> Invocation;
  
  // OnlyLocalDecls - when true, walking this AST should only visit declarations
  // that come from the AST itself, not from included precompiled headers.
  // FIXME: This is temporary; eventually, CIndex will always do this.
  bool                              OnlyLocalDecls;

  /// \brief Whether to capture any diagnostics produced.
  bool CaptureDiagnostics;
  
  /// \brief Track whether the main file was loaded from an AST or not.
  bool MainFileIsAST;

  /// \brief Whether this AST represents a complete translation unit.
  bool CompleteTranslationUnit;

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

  /// \brief Counter that determines when we want to try building a
  /// precompiled preamble.
  ///
  /// If zero, we will never build a precompiled preamble. Otherwise,
  /// it's treated as a counter that decrements each time we reparse
  /// without the benefit of a precompiled preamble. When it hits 1,
  /// we'll attempt to rebuild the precompiled header. This way, if
  /// building the precompiled preamble fails, we won't try again for
  /// some number of calls.
  unsigned PreambleRebuildCounter;
  
  /// \brief The file in which the precompiled preamble is stored.
  std::string PreambleFile;
  
  /// \brief The contents of the preamble that has been precompiled to
  /// \c PreambleFile.
  std::vector<char> Preamble;

  /// \brief Whether the preamble ends at the start of a new line.
  /// 
  /// Used to inform the lexer as to whether it's starting at the beginning of
  /// a line after skipping the preamble.
  bool PreambleEndsAtStartOfLine;
  
  /// \brief The size of the source buffer that we've reserved for the main 
  /// file within the precompiled preamble.
  unsigned PreambleReservedSize;

  /// \brief Keeps track of the files that were used when computing the 
  /// preamble, with both their buffer size and their modification time.
  ///
  /// If any of the files have changed from one compile to the next,
  /// the preamble must be thrown away.
  llvm::StringMap<std::pair<off_t, time_t> > FilesInPreamble;

  /// \brief When non-NULL, this is the buffer used to store the contents of
  /// the main file when it has been padded for use with the precompiled
  /// preamble.
  llvm::MemoryBuffer *SavedMainFileBuffer;

  /// \brief The number of warnings that occurred while parsing the preamble.
  ///
  /// This value will be used to restore the state of the \c Diagnostic object
  /// when re-using the precompiled preamble. Note that only the
  /// number of warnings matters, since we will not save the preamble
  /// when any errors are present.
  unsigned NumWarningsInPreamble;

  /// \brief The number of diagnostics that were stored when parsing
  /// the precompiled preamble.
  ///
  /// This value is used to determine how many of the stored
  /// diagnostics should be retained when reparsing in the presence of
  /// a precompiled preamble.
  unsigned NumStoredDiagnosticsInPreamble;

  /// \brief The group of timers associated with this translation unit.
  llvm::OwningPtr<llvm::TimerGroup> TimerGroup;  

  /// \brief A list of the PCH ID numbers for each of the top-level
  /// declarations parsed within the precompiled preamble.
  std::vector<pch::DeclID> TopLevelDeclsInPreamble;

  ///
  /// \defgroup CodeCompleteCaching Code-completion caching
  ///
  /// \{
  ///

  /// \brief Whether we should be caching code-completion results.
  bool ShouldCacheCodeCompletionResults;
  
public:
  /// \brief A cached code-completion result, which may be introduced in one of
  /// many different contexts.
  struct CachedCodeCompletionResult {
    /// \brief The code-completion string corresponding to this completion
    /// result.
    CodeCompletionString *Completion;
    
    /// \brief A bitmask that indicates which code-completion contexts should
    /// contain this completion result.
    ///
    /// The bits in the bitmask correspond to the values of 
    /// CodeCompleteContext::Kind. To map from a completion context kind to a 
    /// bit, subtract one from the completion context kind and shift 1 by that
    /// number of bits. Many completions can occur in several different
    /// contexts.
    unsigned ShowInContexts;
    
    /// \brief The priority given to this code-completion result.
    unsigned Priority;
    
    /// \brief The libclang cursor kind corresponding to this code-completion 
    /// result.
    CXCursorKind Kind;
    
    /// \brief The simplified type class for a non-macro completion result.
    SimplifiedTypeClass TypeClass;
    
    /// \brief The type of a non-macro completion result, stored as a unique
    /// integer used by the string map of cached completion types.
    ///
    /// This value will be zero if the type is not known, or a unique value
    /// determined by the formatted type string. Se \c CachedCompletionTypes
    /// for more information.
    unsigned Type;
  };
  
  /// \brief Retrieve the mapping from formatted type names to unique type
  /// identifiers.
  llvm::StringMap<unsigned> &getCachedCompletionTypes() { 
    return CachedCompletionTypes; 
  }
  
private:
  /// \brief The set of cached code-completion results.
  std::vector<CachedCodeCompletionResult> CachedCompletionResults;
  
  /// \brief A mapping from the formatted type name to a unique number for that
  /// type, which is used for type equality comparisons.
  llvm::StringMap<unsigned> CachedCompletionTypes;
  
  /// \brief The number of top-level declarations present the last time we
  /// cached code-completion results.
  ///
  /// The value is used to help detect when we should repopulate the global
  /// completion cache.
  unsigned NumTopLevelDeclsAtLastCompletionCache;

  /// \brief The number of reparses left until we'll consider updating the
  /// code-completion cache.
  ///
  /// This is meant to avoid thrashing during reparsing, by not allowing the
  /// code-completion cache to be updated on every reparse.
  unsigned CacheCodeCompletionCoolDown;
  
  /// \brief Cache any "global" code-completion results, so that we can avoid
  /// recomputing them with each completion.
  void CacheCodeCompletionResults();
  
  /// \brief Clear out and deallocate 
  void ClearCachedCompletionResults();
  
  /// 
  /// \}
  ///
  
  /// \brief The timers we've created from the various parses, reparses, etc.
  /// involved in this translation unit.
  std::vector<llvm::Timer *> Timers;
  
  ASTUnit(const ASTUnit&); // DO NOT IMPLEMENT
  ASTUnit &operator=(const ASTUnit &); // DO NOT IMPLEMENT
  
  explicit ASTUnit(bool MainFileIsAST);

  void CleanTemporaryFiles();
  bool Parse(llvm::MemoryBuffer *OverrideMainBuffer);
  
  std::pair<llvm::MemoryBuffer *, std::pair<unsigned, bool> >
  ComputePreamble(CompilerInvocation &Invocation, 
                  unsigned MaxLines, bool &CreatedBuffer);
  
  llvm::MemoryBuffer *getMainBufferWithPrecompiledPreamble(
                                                     bool AllowRebuild = true,
                                                        unsigned MaxLines = 0);
  void RealizeTopLevelDeclsFromPreamble();

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

  bool hasSema() const { return TheSema; }
  Sema &getSema() const { 
    assert(TheSema && "ASTUnit does not have a Sema object!");
    return *TheSema; 
  }
  
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

  /// \brief Retrieve the maximum PCH level of declarations that a
  /// traversal of the translation unit should consider.
  unsigned getMaxPCHLevel() const;

  void setLastASTLocation(ASTLocation ALoc) { LastLoc = ALoc; }
  ASTLocation getLastASTLocation() const { return LastLoc; }

  typedef std::vector<Decl *>::iterator top_level_iterator;

  top_level_iterator top_level_begin() {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    if (!TopLevelDeclsInPreamble.empty())
      RealizeTopLevelDeclsFromPreamble();
    return TopLevelDecls.begin();
  }

  top_level_iterator top_level_end() {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    if (!TopLevelDeclsInPreamble.empty())
      RealizeTopLevelDeclsFromPreamble();
    return TopLevelDecls.end();
  }

  std::size_t top_level_size() const {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    return TopLevelDeclsInPreamble.size() + TopLevelDecls.size();
  }

  bool top_level_empty() const {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    return TopLevelDeclsInPreamble.empty() && TopLevelDecls.empty();
  }

  /// \brief Add a new top-level declaration.
  void addTopLevelDecl(Decl *D) {
    TopLevelDecls.push_back(D);
  }

  /// \brief Add a new top-level declaration, identified by its ID in
  /// the precompiled preamble.
  void addTopLevelDeclFromPreamble(pch::DeclID D) {
    TopLevelDeclsInPreamble.push_back(D);
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

  typedef std::vector<CachedCodeCompletionResult>::iterator
    cached_completion_iterator;
  
  cached_completion_iterator cached_completion_begin() {
    return CachedCompletionResults.begin();
  }

  cached_completion_iterator cached_completion_end() {
    return CachedCompletionResults.end();
  }

  unsigned cached_completion_size() const { 
    return CachedCompletionResults.size(); 
  }
  
  /// \brief Whether this AST represents a complete translation unit.
  ///
  /// If false, this AST is only a partial translation unit, e.g., one
  /// that might still be used as a precompiled header or preamble.
  bool isCompleteTranslationUnit() const { return CompleteTranslationUnit; }

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
                                             bool CaptureDiagnostics = false,
                                             bool PrecompilePreamble = false,
                                          bool CompleteTranslationUnit = true,
                                       bool CacheCodeCompletionResults = false);

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
                                      bool CaptureDiagnostics = false,
                                      bool PrecompilePreamble = false,
                                      bool CompleteTranslationUnit = true,
                                      bool CacheCodeCompletionResults = false);
  
  /// \brief Reparse the source files using the same command-line options that
  /// were originally used to produce this translation unit.
  ///
  /// \returns True if a failure occurred that causes the ASTUnit not to
  /// contain any translation-unit information, false otherwise.  
  bool Reparse(RemappedFile *RemappedFiles = 0,
               unsigned NumRemappedFiles = 0);

  /// \brief Perform code completion at the given file, line, and
  /// column within this translation unit.
  ///
  /// \param File The file in which code completion will occur.
  ///
  /// \param Line The line at which code completion will occur.
  ///
  /// \param Column The column at which code completion will occur.
  ///
  /// \param IncludeMacros Whether to include macros in the code-completion 
  /// results.
  ///
  /// \param IncludeCodePatterns Whether to include code patterns (such as a 
  /// for loop) in the code-completion results.
  ///
  /// FIXME: The Diag, LangOpts, SourceMgr, FileMgr, and
  /// StoredDiagnostics parameters are all disgusting hacks. They will
  /// go away.
  void CodeComplete(llvm::StringRef File, unsigned Line, unsigned Column,
                    RemappedFile *RemappedFiles, unsigned NumRemappedFiles,
                    bool IncludeMacros, bool IncludeCodePatterns,
                    CodeCompleteConsumer &Consumer,
                    Diagnostic &Diag, LangOptions &LangOpts,
                    SourceManager &SourceMgr, FileManager &FileMgr,
                    llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiagnostics);

  /// \brief Save this translation unit to a file with the given name.
  ///
  /// \returns True if an error occurred, false otherwise.
  bool Save(llvm::StringRef File);
};

} // namespace clang

#endif
