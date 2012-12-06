//===- CIndexHigh.cpp - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"
#include "CIndexDiagnostic.h"
#include "CIndexer.h"
#include "CXCursor.h"
#include "CXSourceLocation.h"
#include "CXString.h"
#include "CXTranslationUnit.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPConditionalDirectiveRecord.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Sema/SemaConsumer.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/MutexGuard.h"

using namespace clang;
using namespace cxstring;
using namespace cxtu;
using namespace cxindex;

static void indexDiagnostics(CXTranslationUnit TU, IndexingContext &IdxCtx);

namespace {

//===----------------------------------------------------------------------===//
// Skip Parsed Bodies
//===----------------------------------------------------------------------===//

#ifdef LLVM_ON_WIN32

// FIXME: On windows it is disabled since current implementation depends on
// file inodes.

class SessionSkipBodyData { };

class TUSkipBodyControl {
public:
  TUSkipBodyControl(SessionSkipBodyData &sessionData,
                    PPConditionalDirectiveRecord &ppRec) { }
  bool isParsed(SourceLocation Loc, FileID FID, const FileEntry *FE) {
    return false;
  }
  void finished() { }
};

#else

/// \brief A "region" in source code identified by the file/offset of the
/// preprocessor conditional directive that it belongs to.
/// Multiple, non-consecutive ranges can be parts of the same region.
///
/// As an example of different regions separated by preprocessor directives:
///
/// \code
///   #1
/// #ifdef BLAH
///   #2
/// #ifdef CAKE
///   #3
/// #endif
///   #2
/// #endif
///   #1
/// \endcode
///
/// There are 3 regions, with non-consecutive parts:
///   #1 is identified as the beginning of the file
///   #2 is identified as the location of "#ifdef BLAH"
///   #3 is identified as the location of "#ifdef CAKE"
///
class PPRegion {
  ino_t ino;
  time_t ModTime;
  dev_t dev;
  unsigned Offset;
public:
  PPRegion() : ino(), ModTime(), dev(), Offset() {}
  PPRegion(dev_t dev, ino_t ino, unsigned offset, time_t modTime)
    : ino(ino), ModTime(modTime), dev(dev), Offset(offset) {}

  ino_t getIno() const { return ino; }
  dev_t getDev() const { return dev; }
  unsigned getOffset() const { return Offset; }
  time_t getModTime() const { return ModTime; }

  bool isInvalid() const { return *this == PPRegion(); }

  friend bool operator==(const PPRegion &lhs, const PPRegion &rhs) {
    return lhs.dev == rhs.dev && lhs.ino == rhs.ino &&
        lhs.Offset == rhs.Offset && lhs.ModTime == rhs.ModTime;
  }
};

typedef llvm::DenseSet<PPRegion> PPRegionSetTy;

} // end anonymous namespace

namespace llvm {
  template <> struct isPodLike<PPRegion> {
    static const bool value = true;
  };

  template <>
  struct DenseMapInfo<PPRegion> {
    static inline PPRegion getEmptyKey() {
      return PPRegion(0, 0, unsigned(-1), 0);
    }
    static inline PPRegion getTombstoneKey() {
      return PPRegion(0, 0, unsigned(-2), 0);
    }

    static unsigned getHashValue(const PPRegion &S) {
      llvm::FoldingSetNodeID ID;
      ID.AddInteger(S.getIno());
      ID.AddInteger(S.getDev());
      ID.AddInteger(S.getOffset());
      ID.AddInteger(S.getModTime());
      return ID.ComputeHash();
    }

    static bool isEqual(const PPRegion &LHS, const PPRegion &RHS) {
      return LHS == RHS;
    }
  };
}

namespace {

class SessionSkipBodyData {
  llvm::sys::Mutex Mux;
  PPRegionSetTy ParsedRegions;

public:
  SessionSkipBodyData() : Mux(/*recursive=*/false) {}
  ~SessionSkipBodyData() {
    //llvm::errs() << "RegionData: " << Skipped.size() << " - " << Skipped.getMemorySize() << "\n";
  }

  void copyTo(PPRegionSetTy &Set) {
    llvm::MutexGuard MG(Mux);
    Set = ParsedRegions;
  }

  void update(ArrayRef<PPRegion> Regions) {
    llvm::MutexGuard MG(Mux);
    ParsedRegions.insert(Regions.begin(), Regions.end());
  }
};

class TUSkipBodyControl {
  SessionSkipBodyData &SessionData;
  PPConditionalDirectiveRecord &PPRec;
  Preprocessor &PP;

  PPRegionSetTy ParsedRegions;
  SmallVector<PPRegion, 32> NewParsedRegions;
  PPRegion LastRegion;
  bool LastIsParsed;

public:
  TUSkipBodyControl(SessionSkipBodyData &sessionData,
                    PPConditionalDirectiveRecord &ppRec,
                    Preprocessor &pp)
    : SessionData(sessionData), PPRec(ppRec), PP(pp) {
    SessionData.copyTo(ParsedRegions);
  }

  bool isParsed(SourceLocation Loc, FileID FID, const FileEntry *FE) {
    PPRegion region = getRegion(Loc, FID, FE);
    if (region.isInvalid())
      return false;

    // Check common case, consecutive functions in the same region.
    if (LastRegion == region)
      return LastIsParsed;

    LastRegion = region;
    LastIsParsed = ParsedRegions.count(region);
    if (!LastIsParsed)
      NewParsedRegions.push_back(region);
    return LastIsParsed;
  }

  void finished() {
    SessionData.update(NewParsedRegions);
  }

private:
  PPRegion getRegion(SourceLocation Loc, FileID FID, const FileEntry *FE) {
    SourceLocation RegionLoc = PPRec.findConditionalDirectiveRegionLoc(Loc);
    if (RegionLoc.isInvalid()) {
      if (isParsedOnceInclude(FE))
        return PPRegion(FE->getDevice(), FE->getInode(), 0,
                        FE->getModificationTime());
      return PPRegion();
    }

    const SourceManager &SM = PPRec.getSourceManager();
    assert(RegionLoc.isFileID());
    FileID RegionFID;
    unsigned RegionOffset;
    llvm::tie(RegionFID, RegionOffset) = SM.getDecomposedLoc(RegionLoc);

    if (RegionFID != FID) {
      if (isParsedOnceInclude(FE))
        return PPRegion(FE->getDevice(), FE->getInode(), 0,
                        FE->getModificationTime());
      return PPRegion();
    }

    return PPRegion(FE->getDevice(), FE->getInode(), RegionOffset,
                    FE->getModificationTime());
  }

  bool isParsedOnceInclude(const FileEntry *FE) {
    return PP.getHeaderSearchInfo().isFileMultipleIncludeGuarded(FE);
  }
};

#endif

//===----------------------------------------------------------------------===//
// IndexPPCallbacks
//===----------------------------------------------------------------------===//

class IndexPPCallbacks : public PPCallbacks {
  Preprocessor &PP;
  IndexingContext &IndexCtx;
  bool IsMainFileEntered;

public:
  IndexPPCallbacks(Preprocessor &PP, IndexingContext &indexCtx)
    : PP(PP), IndexCtx(indexCtx), IsMainFileEntered(false) { }

  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                          SrcMgr::CharacteristicKind FileType, FileID PrevFID) {
    if (IsMainFileEntered)
      return;

    SourceManager &SM = PP.getSourceManager();
    SourceLocation MainFileLoc = SM.getLocForStartOfFile(SM.getMainFileID());

    if (Loc == MainFileLoc && Reason == PPCallbacks::EnterFile) {
      IsMainFileEntered = true;
      IndexCtx.enteredMainFile(SM.getFileEntryForID(SM.getMainFileID()));
    }
  }

  virtual void InclusionDirective(SourceLocation HashLoc,
                                  const Token &IncludeTok,
                                  StringRef FileName,
                                  bool IsAngled,
                                  CharSourceRange FilenameRange,
                                  const FileEntry *File,
                                  StringRef SearchPath,
                                  StringRef RelativePath,
                                  const Module *Imported) {
    bool isImport = (IncludeTok.is(tok::identifier) &&
            IncludeTok.getIdentifierInfo()->getPPKeywordID() == tok::pp_import);
    IndexCtx.ppIncludedFile(HashLoc, FileName, File, isImport, IsAngled,
                            Imported);
  }

  /// MacroDefined - This hook is called whenever a macro definition is seen.
  virtual void MacroDefined(const Token &Id, const MacroInfo *MI) {
  }

  /// MacroUndefined - This hook is called whenever a macro #undef is seen.
  /// MI is released immediately following this callback.
  virtual void MacroUndefined(const Token &MacroNameTok, const MacroInfo *MI) {
  }

  /// MacroExpands - This is called by when a macro invocation is found.
  virtual void MacroExpands(const Token &MacroNameTok, const MacroInfo* MI,
                            SourceRange Range) {
  }

  /// SourceRangeSkipped - This hook is called when a source range is skipped.
  /// \param Range The SourceRange that was skipped. The range begins at the
  /// #if/#else directive and ends after the #endif/#else directive.
  virtual void SourceRangeSkipped(SourceRange Range) {
  }
};

//===----------------------------------------------------------------------===//
// IndexingConsumer
//===----------------------------------------------------------------------===//

class IndexingConsumer : public ASTConsumer {
  IndexingContext &IndexCtx;
  TUSkipBodyControl *SKCtrl;

public:
  IndexingConsumer(IndexingContext &indexCtx, TUSkipBodyControl *skCtrl)
    : IndexCtx(indexCtx), SKCtrl(skCtrl) { }

  // ASTConsumer Implementation

  virtual void Initialize(ASTContext &Context) {
    IndexCtx.setASTContext(Context);
    IndexCtx.startedTranslationUnit();
  }

  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    if (SKCtrl)
      SKCtrl->finished();
  }

  virtual bool HandleTopLevelDecl(DeclGroupRef DG) {
    IndexCtx.indexDeclGroupRef(DG);
    return !IndexCtx.shouldAbort();
  }

  /// \brief Handle the specified top-level declaration that occurred inside
  /// and ObjC container.
  virtual void HandleTopLevelDeclInObjCContainer(DeclGroupRef D) {
    // They will be handled after the interface is seen first.
    IndexCtx.addTUDeclInObjCContainer(D);
  }

  /// \brief This is called by the AST reader when deserializing things.
  /// The default implementation forwards to HandleTopLevelDecl but we don't
  /// care about them when indexing, so have an empty definition.
  virtual void HandleInterestingDecl(DeclGroupRef D) {}

  virtual void HandleTagDeclDefinition(TagDecl *D) {
    if (!IndexCtx.shouldIndexImplicitTemplateInsts())
      return;

    if (IndexCtx.isTemplateImplicitInstantiation(D))
      IndexCtx.indexDecl(D);
  }

  virtual void HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) {
    if (!IndexCtx.shouldIndexImplicitTemplateInsts())
      return;

    IndexCtx.indexDecl(D);
  }

  virtual bool shouldSkipFunctionBody(Decl *D) {
    if (!SKCtrl) {
      // Always skip bodies.
      return true;
    }

    const SourceManager &SM = IndexCtx.getASTContext().getSourceManager();
    SourceLocation Loc = D->getLocation();
    if (Loc.isMacroID())
      return false;
    if (SM.isInSystemHeader(Loc))
      return true; // always skip bodies from system headers.

    FileID FID;
    unsigned Offset;
    llvm::tie(FID, Offset) = SM.getDecomposedLoc(Loc);
    // Don't skip bodies from main files; this may be revisited.
    if (SM.getMainFileID() == FID)
      return false;
    const FileEntry *FE = SM.getFileEntryForID(FID);
    if (!FE)
      return false;

    return SKCtrl->isParsed(Loc, FID, FE);
  }
};

//===----------------------------------------------------------------------===//
// CaptureDiagnosticConsumer
//===----------------------------------------------------------------------===//

class CaptureDiagnosticConsumer : public DiagnosticConsumer {
  SmallVector<StoredDiagnostic, 4> Errors;
public:

  virtual void HandleDiagnostic(DiagnosticsEngine::Level level,
                                const Diagnostic &Info) {
    if (level >= DiagnosticsEngine::Error)
      Errors.push_back(StoredDiagnostic(level, Info));
  }

  DiagnosticConsumer *clone(DiagnosticsEngine &Diags) const {
    return new IgnoringDiagConsumer();
  }
};

//===----------------------------------------------------------------------===//
// IndexingFrontendAction
//===----------------------------------------------------------------------===//

class IndexingFrontendAction : public ASTFrontendAction {
  IndexingContext IndexCtx;
  CXTranslationUnit CXTU;

  SessionSkipBodyData *SKData;
  OwningPtr<TUSkipBodyControl> SKCtrl;

public:
  IndexingFrontendAction(CXClientData clientData,
                         IndexerCallbacks &indexCallbacks,
                         unsigned indexOptions,
                         CXTranslationUnit cxTU,
                         SessionSkipBodyData *skData)
    : IndexCtx(clientData, indexCallbacks, indexOptions, cxTU),
      CXTU(cxTU), SKData(skData) { }

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile) {
    PreprocessorOptions &PPOpts = CI.getPreprocessorOpts();

    if (!PPOpts.ImplicitPCHInclude.empty()) {
      IndexCtx.importedPCH(
                        CI.getFileManager().getFile(PPOpts.ImplicitPCHInclude));
    }

    IndexCtx.setASTContext(CI.getASTContext());
    Preprocessor &PP = CI.getPreprocessor();
    PP.addPPCallbacks(new IndexPPCallbacks(PP, IndexCtx));
    IndexCtx.setPreprocessor(PP);

    if (SKData) {
      PPConditionalDirectiveRecord *
        PPRec = new PPConditionalDirectiveRecord(PP.getSourceManager());
      PP.addPPCallbacks(PPRec);
      SKCtrl.reset(new TUSkipBodyControl(*SKData, *PPRec, PP));
    }

    return new IndexingConsumer(IndexCtx, SKCtrl.get());
  }

  virtual void EndSourceFileAction() {
    indexDiagnostics(CXTU, IndexCtx);
  }

  virtual TranslationUnitKind getTranslationUnitKind() {
    if (IndexCtx.shouldIndexImplicitTemplateInsts())
      return TU_Complete;
    else
      return TU_Prefix;
  }
  virtual bool hasCodeCompletionSupport() const { return false; }
};

//===----------------------------------------------------------------------===//
// clang_indexSourceFileUnit Implementation
//===----------------------------------------------------------------------===//

struct IndexSessionData {
  CXIndex CIdx;
  OwningPtr<SessionSkipBodyData> SkipBodyData;

  explicit IndexSessionData(CXIndex cIdx)
    : CIdx(cIdx), SkipBodyData(new SessionSkipBodyData) {}
};

struct IndexSourceFileInfo {
  CXIndexAction idxAction;
  CXClientData client_data;
  IndexerCallbacks *index_callbacks;
  unsigned index_callbacks_size;
  unsigned index_options;
  const char *source_filename;
  const char *const *command_line_args;
  int num_command_line_args;
  struct CXUnsavedFile *unsaved_files;
  unsigned num_unsaved_files;
  CXTranslationUnit *out_TU;
  unsigned TU_options;
  int result;
};

struct MemBufferOwner {
  SmallVector<const llvm::MemoryBuffer *, 8> Buffers;
  
  ~MemBufferOwner() {
    for (SmallVectorImpl<const llvm::MemoryBuffer *>::iterator
           I = Buffers.begin(), E = Buffers.end(); I != E; ++I)
      delete *I;
  }
};

} // anonymous namespace

static void clang_indexSourceFile_Impl(void *UserData) {
  IndexSourceFileInfo *ITUI =
    static_cast<IndexSourceFileInfo*>(UserData);
  CXIndexAction cxIdxAction = ITUI->idxAction;
  CXClientData client_data = ITUI->client_data;
  IndexerCallbacks *client_index_callbacks = ITUI->index_callbacks;
  unsigned index_callbacks_size = ITUI->index_callbacks_size;
  unsigned index_options = ITUI->index_options;
  const char *source_filename = ITUI->source_filename;
  const char * const *command_line_args = ITUI->command_line_args;
  int num_command_line_args = ITUI->num_command_line_args;
  struct CXUnsavedFile *unsaved_files = ITUI->unsaved_files;
  unsigned num_unsaved_files = ITUI->num_unsaved_files;
  CXTranslationUnit *out_TU  = ITUI->out_TU;
  unsigned TU_options = ITUI->TU_options;
  ITUI->result = 1; // init as error.
  
  if (out_TU)
    *out_TU = 0;
  bool requestedToGetTU = (out_TU != 0); 

  if (!cxIdxAction)
    return;
  if (!client_index_callbacks || index_callbacks_size == 0)
    return;

  IndexerCallbacks CB;
  memset(&CB, 0, sizeof(CB));
  unsigned ClientCBSize = index_callbacks_size < sizeof(CB)
                                  ? index_callbacks_size : sizeof(CB);
  memcpy(&CB, client_index_callbacks, ClientCBSize);

  IndexSessionData *IdxSession = static_cast<IndexSessionData *>(cxIdxAction);
  CIndexer *CXXIdx = static_cast<CIndexer *>(IdxSession->CIdx);

  if (CXXIdx->isOptEnabled(CXGlobalOpt_ThreadBackgroundPriorityForIndexing))
    setThreadBackgroundPriority();

  CaptureDiagnosticConsumer *CaptureDiag = new CaptureDiagnosticConsumer();

  // Configure the diagnostics.
  IntrusiveRefCntPtr<DiagnosticsEngine>
    Diags(CompilerInstance::createDiagnostics(new DiagnosticOptions,
                                              num_command_line_args,
                                              command_line_args,
                                              CaptureDiag,
                                              /*ShouldOwnClient=*/true,
                                              /*ShouldCloneClient=*/false));

  // Recover resources if we crash before exiting this function.
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.getPtr());
  
  OwningPtr<std::vector<const char *> >
    Args(new std::vector<const char*>());

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<std::vector<const char*> >
    ArgsCleanup(Args.get());
  
  Args->insert(Args->end(), command_line_args,
               command_line_args + num_command_line_args);

  // The 'source_filename' argument is optional.  If the caller does not
  // specify it then it is assumed that the source file is specified
  // in the actual argument list.
  // Put the source file after command_line_args otherwise if '-x' flag is
  // present it will be unused.
  if (source_filename)
    Args->push_back(source_filename);
  
  IntrusiveRefCntPtr<CompilerInvocation>
    CInvok(createInvocationFromCommandLine(*Args, Diags));

  if (!CInvok)
    return;

  // Recover resources if we crash before exiting this function.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInvocation,
    llvm::CrashRecoveryContextReleaseRefCleanup<CompilerInvocation> >
    CInvokCleanup(CInvok.getPtr());

  if (CInvok->getFrontendOpts().Inputs.empty())
    return;

  OwningPtr<MemBufferOwner> BufOwner(new MemBufferOwner());

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<MemBufferOwner>
    BufOwnerCleanup(BufOwner.get());

  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    StringRef Data(unsaved_files[I].Contents, unsaved_files[I].Length);
    const llvm::MemoryBuffer *Buffer
      = llvm::MemoryBuffer::getMemBufferCopy(Data, unsaved_files[I].Filename);
    CInvok->getPreprocessorOpts().addRemappedFile(unsaved_files[I].Filename, Buffer);
    BufOwner->Buffers.push_back(Buffer);
  }

  // Since libclang is primarily used by batch tools dealing with
  // (often very broken) source code, where spell-checking can have a
  // significant negative impact on performance (particularly when 
  // precompiled headers are involved), we disable it.
  CInvok->getLangOpts()->SpellChecking = false;

  if (index_options & CXIndexOpt_SuppressWarnings)
    CInvok->getDiagnosticOpts().IgnoreWarnings = true;

  ASTUnit *Unit = ASTUnit::create(CInvok.getPtr(), Diags,
                                  /*CaptureDiagnostics=*/true,
                                  /*UserFilesAreVolatile=*/true);
  OwningPtr<CXTUOwner> CXTU(new CXTUOwner(MakeCXTranslationUnit(CXXIdx, Unit)));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CXTUOwner>
    CXTUCleanup(CXTU.get());

  // Enable the skip-parsed-bodies optimization only for C++; this may be
  // revisited.
  bool SkipBodies = (index_options & CXIndexOpt_SkipParsedBodiesInSession) &&
      CInvok->getLangOpts()->CPlusPlus;
  if (SkipBodies)
    CInvok->getFrontendOpts().SkipFunctionBodies = true;

  OwningPtr<IndexingFrontendAction> IndexAction;
  IndexAction.reset(new IndexingFrontendAction(client_data, CB,
                                               index_options, CXTU->getTU(),
                              SkipBodies ? IdxSession->SkipBodyData.get() : 0));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<IndexingFrontendAction>
    IndexActionCleanup(IndexAction.get());

  bool Persistent = requestedToGetTU;
  bool OnlyLocalDecls = false;
  bool PrecompilePreamble = false;
  bool CacheCodeCompletionResults = false;
  PreprocessorOptions &PPOpts = CInvok->getPreprocessorOpts(); 
  PPOpts.AllowPCHWithCompilerErrors = true;

  if (requestedToGetTU) {
    OnlyLocalDecls = CXXIdx->getOnlyLocalDecls();
    PrecompilePreamble = TU_options & CXTranslationUnit_PrecompiledPreamble;
    // FIXME: Add a flag for modules.
    CacheCodeCompletionResults
      = TU_options & CXTranslationUnit_CacheCompletionResults;
  }

  if (TU_options & CXTranslationUnit_DetailedPreprocessingRecord) {
    PPOpts.DetailedRecord = true;
  }

  if (!requestedToGetTU && !CInvok->getLangOpts()->Modules)
    PPOpts.DetailedRecord = false;

  DiagnosticErrorTrap DiagTrap(*Diags);
  bool Success = ASTUnit::LoadFromCompilerInvocationAction(CInvok.getPtr(), Diags,
                                                       IndexAction.get(),
                                                       Unit,
                                                       Persistent,
                                                CXXIdx->getClangResourcesPath(),
                                                       OnlyLocalDecls,
                                                    /*CaptureDiagnostics=*/true,
                                                       PrecompilePreamble,
                                                    CacheCodeCompletionResults,
                                 /*IncludeBriefCommentsInCodeCompletion=*/false,
                                                 /*UserFilesAreVolatile=*/true);
  if (DiagTrap.hasErrorOccurred() && CXXIdx->getDisplayDiagnostics())
    printDiagsToStderr(Unit);

  if (!Success)
    return;

  if (out_TU)
    *out_TU = CXTU->takeTU();

  ITUI->result = 0; // success.
}

//===----------------------------------------------------------------------===//
// clang_indexTranslationUnit Implementation
//===----------------------------------------------------------------------===//

namespace {

struct IndexTranslationUnitInfo {
  CXIndexAction idxAction;
  CXClientData client_data;
  IndexerCallbacks *index_callbacks;
  unsigned index_callbacks_size;
  unsigned index_options;
  CXTranslationUnit TU;
  int result;
};

} // anonymous namespace

static void indexPreprocessingRecord(ASTUnit &Unit, IndexingContext &IdxCtx) {
  Preprocessor &PP = Unit.getPreprocessor();
  if (!PP.getPreprocessingRecord())
    return;

  // FIXME: Only deserialize inclusion directives.

  PreprocessingRecord::iterator I, E;
  llvm::tie(I, E) = Unit.getLocalPreprocessingEntities();

  bool isModuleFile = Unit.isModuleFile();
  for (; I != E; ++I) {
    PreprocessedEntity *PPE = *I;

    if (InclusionDirective *ID = dyn_cast<InclusionDirective>(PPE)) {
      SourceLocation Loc = ID->getSourceRange().getBegin();
      // Modules have synthetic main files as input, give an invalid location
      // if the location points to such a file.
      if (isModuleFile && Unit.isInMainFileID(Loc))
        Loc = SourceLocation();
      IdxCtx.ppIncludedFile(Loc, ID->getFileName(),
                            ID->getFile(),
                            ID->getKind() == InclusionDirective::Import,
                            !ID->wasInQuotes(), ID->importedModule());
    }
  }
}

static bool topLevelDeclVisitor(void *context, const Decl *D) {
  IndexingContext &IdxCtx = *static_cast<IndexingContext*>(context);
  IdxCtx.indexTopLevelDecl(D);
  if (IdxCtx.shouldAbort())
    return false;
  return true;
}

static void indexTranslationUnit(ASTUnit &Unit, IndexingContext &IdxCtx) {
  Unit.visitLocalTopLevelDecls(&IdxCtx, topLevelDeclVisitor);
}

static void indexDiagnostics(CXTranslationUnit TU, IndexingContext &IdxCtx) {
  if (!IdxCtx.hasDiagnosticCallback())
    return;

  CXDiagnosticSetImpl *DiagSet = cxdiag::lazyCreateDiags(TU);
  IdxCtx.handleDiagnosticSet(DiagSet);
}

static void clang_indexTranslationUnit_Impl(void *UserData) {
  IndexTranslationUnitInfo *ITUI =
    static_cast<IndexTranslationUnitInfo*>(UserData);
  CXTranslationUnit TU = ITUI->TU;
  CXClientData client_data = ITUI->client_data;
  IndexerCallbacks *client_index_callbacks = ITUI->index_callbacks;
  unsigned index_callbacks_size = ITUI->index_callbacks_size;
  unsigned index_options = ITUI->index_options;
  ITUI->result = 1; // init as error.

  if (!TU)
    return;
  if (!client_index_callbacks || index_callbacks_size == 0)
    return;

  CIndexer *CXXIdx = (CIndexer*)TU->CIdx;
  if (CXXIdx->isOptEnabled(CXGlobalOpt_ThreadBackgroundPriorityForIndexing))
    setThreadBackgroundPriority();

  IndexerCallbacks CB;
  memset(&CB, 0, sizeof(CB));
  unsigned ClientCBSize = index_callbacks_size < sizeof(CB)
                                  ? index_callbacks_size : sizeof(CB);
  memcpy(&CB, client_index_callbacks, ClientCBSize);

  OwningPtr<IndexingContext> IndexCtx;
  IndexCtx.reset(new IndexingContext(client_data, CB, index_options, TU));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<IndexingContext>
    IndexCtxCleanup(IndexCtx.get());

  OwningPtr<IndexingConsumer> IndexConsumer;
  IndexConsumer.reset(new IndexingConsumer(*IndexCtx, 0));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<IndexingConsumer>
    IndexConsumerCleanup(IndexConsumer.get());

  ASTUnit *Unit = static_cast<ASTUnit *>(TU->TUData);
  if (!Unit)
    return;

  ASTUnit::ConcurrencyCheck Check(*Unit);

  if (const FileEntry *PCHFile = Unit->getPCHFile())
    IndexCtx->importedPCH(PCHFile);

  FileManager &FileMgr = Unit->getFileManager();

  if (Unit->getOriginalSourceFileName().empty())
    IndexCtx->enteredMainFile(0);
  else
    IndexCtx->enteredMainFile(FileMgr.getFile(Unit->getOriginalSourceFileName()));

  IndexConsumer->Initialize(Unit->getASTContext());

  indexPreprocessingRecord(*Unit, *IndexCtx);
  indexTranslationUnit(*Unit, *IndexCtx);
  indexDiagnostics(TU, *IndexCtx);

  ITUI->result = 0;
}

//===----------------------------------------------------------------------===//
// libclang public APIs.
//===----------------------------------------------------------------------===//

extern "C" {

int clang_index_isEntityObjCContainerKind(CXIdxEntityKind K) {
  return CXIdxEntity_ObjCClass <= K && K <= CXIdxEntity_ObjCCategory;
}

const CXIdxObjCContainerDeclInfo *
clang_index_getObjCContainerDeclInfo(const CXIdxDeclInfo *DInfo) {
  if (!DInfo)
    return 0;

  const DeclInfo *DI = static_cast<const DeclInfo *>(DInfo);
  if (const ObjCContainerDeclInfo *
        ContInfo = dyn_cast<ObjCContainerDeclInfo>(DI))
    return &ContInfo->ObjCContDeclInfo;

  return 0;
}

const CXIdxObjCInterfaceDeclInfo *
clang_index_getObjCInterfaceDeclInfo(const CXIdxDeclInfo *DInfo) {
  if (!DInfo)
    return 0;

  const DeclInfo *DI = static_cast<const DeclInfo *>(DInfo);
  if (const ObjCInterfaceDeclInfo *
        InterInfo = dyn_cast<ObjCInterfaceDeclInfo>(DI))
    return &InterInfo->ObjCInterDeclInfo;

  return 0;
}

const CXIdxObjCCategoryDeclInfo *
clang_index_getObjCCategoryDeclInfo(const CXIdxDeclInfo *DInfo){
  if (!DInfo)
    return 0;

  const DeclInfo *DI = static_cast<const DeclInfo *>(DInfo);
  if (const ObjCCategoryDeclInfo *
        CatInfo = dyn_cast<ObjCCategoryDeclInfo>(DI))
    return &CatInfo->ObjCCatDeclInfo;

  return 0;
}

const CXIdxObjCProtocolRefListInfo *
clang_index_getObjCProtocolRefListInfo(const CXIdxDeclInfo *DInfo) {
  if (!DInfo)
    return 0;

  const DeclInfo *DI = static_cast<const DeclInfo *>(DInfo);
  
  if (const ObjCInterfaceDeclInfo *
        InterInfo = dyn_cast<ObjCInterfaceDeclInfo>(DI))
    return InterInfo->ObjCInterDeclInfo.protocols;
  
  if (const ObjCProtocolDeclInfo *
        ProtInfo = dyn_cast<ObjCProtocolDeclInfo>(DI))
    return &ProtInfo->ObjCProtoRefListInfo;

  if (const ObjCCategoryDeclInfo *CatInfo = dyn_cast<ObjCCategoryDeclInfo>(DI))
    return CatInfo->ObjCCatDeclInfo.protocols;

  return 0;
}

const CXIdxObjCPropertyDeclInfo *
clang_index_getObjCPropertyDeclInfo(const CXIdxDeclInfo *DInfo) {
  if (!DInfo)
    return 0;

  const DeclInfo *DI = static_cast<const DeclInfo *>(DInfo);
  if (const ObjCPropertyDeclInfo *PropInfo = dyn_cast<ObjCPropertyDeclInfo>(DI))
    return &PropInfo->ObjCPropDeclInfo;

  return 0;
}

const CXIdxIBOutletCollectionAttrInfo *
clang_index_getIBOutletCollectionAttrInfo(const CXIdxAttrInfo *AInfo) {
  if (!AInfo)
    return 0;

  const AttrInfo *DI = static_cast<const AttrInfo *>(AInfo);
  if (const IBOutletCollectionInfo *
        IBInfo = dyn_cast<IBOutletCollectionInfo>(DI))
    return &IBInfo->IBCollInfo;

  return 0;
}

const CXIdxCXXClassDeclInfo *
clang_index_getCXXClassDeclInfo(const CXIdxDeclInfo *DInfo) {
  if (!DInfo)
    return 0;

  const DeclInfo *DI = static_cast<const DeclInfo *>(DInfo);
  if (const CXXClassDeclInfo *ClassInfo = dyn_cast<CXXClassDeclInfo>(DI))
    return &ClassInfo->CXXClassInfo;

  return 0;
}

CXIdxClientContainer
clang_index_getClientContainer(const CXIdxContainerInfo *info) {
  if (!info)
    return 0;
  const ContainerInfo *Container = static_cast<const ContainerInfo *>(info);
  return Container->IndexCtx->getClientContainerForDC(Container->DC);
}

void clang_index_setClientContainer(const CXIdxContainerInfo *info,
                                    CXIdxClientContainer client) {
  if (!info)
    return;
  const ContainerInfo *Container = static_cast<const ContainerInfo *>(info);
  Container->IndexCtx->addContainerInMap(Container->DC, client);
}

CXIdxClientEntity clang_index_getClientEntity(const CXIdxEntityInfo *info) {
  if (!info)
    return 0;
  const EntityInfo *Entity = static_cast<const EntityInfo *>(info);
  return Entity->IndexCtx->getClientEntity(Entity->Dcl);
}

void clang_index_setClientEntity(const CXIdxEntityInfo *info,
                                 CXIdxClientEntity client) {
  if (!info)
    return;
  const EntityInfo *Entity = static_cast<const EntityInfo *>(info);
  Entity->IndexCtx->setClientEntity(Entity->Dcl, client);
}

CXIndexAction clang_IndexAction_create(CXIndex CIdx) {
  return new IndexSessionData(CIdx);
}

void clang_IndexAction_dispose(CXIndexAction idxAction) {
  if (idxAction)
    delete static_cast<IndexSessionData *>(idxAction);
}

int clang_indexSourceFile(CXIndexAction idxAction,
                          CXClientData client_data,
                          IndexerCallbacks *index_callbacks,
                          unsigned index_callbacks_size,
                          unsigned index_options,
                          const char *source_filename,
                          const char * const *command_line_args,
                          int num_command_line_args,
                          struct CXUnsavedFile *unsaved_files,
                          unsigned num_unsaved_files,
                          CXTranslationUnit *out_TU,
                          unsigned TU_options) {

  IndexSourceFileInfo ITUI = { idxAction, client_data, index_callbacks,
                               index_callbacks_size, index_options,
                               source_filename, command_line_args,
                               num_command_line_args, unsaved_files,
                               num_unsaved_files, out_TU, TU_options, 0 };

  if (getenv("LIBCLANG_NOTHREADS")) {
    clang_indexSourceFile_Impl(&ITUI);
    return ITUI.result;
  }

  llvm::CrashRecoveryContext CRC;

  if (!RunSafely(CRC, clang_indexSourceFile_Impl, &ITUI)) {
    fprintf(stderr, "libclang: crash detected during indexing source file: {\n");
    fprintf(stderr, "  'source_filename' : '%s'\n", source_filename);
    fprintf(stderr, "  'command_line_args' : [");
    for (int i = 0; i != num_command_line_args; ++i) {
      if (i)
        fprintf(stderr, ", ");
      fprintf(stderr, "'%s'", command_line_args[i]);
    }
    fprintf(stderr, "],\n");
    fprintf(stderr, "  'unsaved_files' : [");
    for (unsigned i = 0; i != num_unsaved_files; ++i) {
      if (i)
        fprintf(stderr, ", ");
      fprintf(stderr, "('%s', '...', %ld)", unsaved_files[i].Filename,
              unsaved_files[i].Length);
    }
    fprintf(stderr, "],\n");
    fprintf(stderr, "  'options' : %d,\n", TU_options);
    fprintf(stderr, "}\n");
    
    return 1;
  } else if (getenv("LIBCLANG_RESOURCE_USAGE")) {
    if (out_TU)
      PrintLibclangResourceUsage(*out_TU);
  }
  
  return ITUI.result;
}

int clang_indexTranslationUnit(CXIndexAction idxAction,
                               CXClientData client_data,
                               IndexerCallbacks *index_callbacks,
                               unsigned index_callbacks_size,
                               unsigned index_options,
                               CXTranslationUnit TU) {

  IndexTranslationUnitInfo ITUI = { idxAction, client_data, index_callbacks,
                                    index_callbacks_size, index_options, TU,
                                    0 };

  if (getenv("LIBCLANG_NOTHREADS")) {
    clang_indexTranslationUnit_Impl(&ITUI);
    return ITUI.result;
  }

  llvm::CrashRecoveryContext CRC;

  if (!RunSafely(CRC, clang_indexTranslationUnit_Impl, &ITUI)) {
    fprintf(stderr, "libclang: crash detected during indexing TU\n");
    
    return 1;
  }

  return ITUI.result;
}

void clang_indexLoc_getFileLocation(CXIdxLoc location,
                                    CXIdxClientFile *indexFile,
                                    CXFile *file,
                                    unsigned *line,
                                    unsigned *column,
                                    unsigned *offset) {
  if (indexFile) *indexFile = 0;
  if (file)   *file = 0;
  if (line)   *line = 0;
  if (column) *column = 0;
  if (offset) *offset = 0;

  SourceLocation Loc = SourceLocation::getFromRawEncoding(location.int_data);
  if (!location.ptr_data[0] || Loc.isInvalid())
    return;

  IndexingContext &IndexCtx =
      *static_cast<IndexingContext*>(location.ptr_data[0]);
  IndexCtx.translateLoc(Loc, indexFile, file, line, column, offset);
}

CXSourceLocation clang_indexLoc_getCXSourceLocation(CXIdxLoc location) {
  SourceLocation Loc = SourceLocation::getFromRawEncoding(location.int_data);
  if (!location.ptr_data[0] || Loc.isInvalid())
    return clang_getNullLocation();

  IndexingContext &IndexCtx =
      *static_cast<IndexingContext*>(location.ptr_data[0]);
  return cxloc::translateSourceLocation(IndexCtx.getASTContext(), Loc);
}

} // end: extern "C"

