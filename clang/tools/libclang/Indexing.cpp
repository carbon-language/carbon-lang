//===- CIndexHigh.cpp - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"
#include "CXCursor.h"
#include "CXSourceLocation.h"
#include "CXTranslationUnit.h"
#include "CXString.h"
#include "CIndexer.h"

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CrashRecoveryContext.h"

using namespace clang;
using namespace cxstring;
using namespace cxtu;
using namespace cxindex;

namespace {

//===----------------------------------------------------------------------===//
// IndexPPCallbacks
//===----------------------------------------------------------------------===//

class IndexPPCallbacks : public PPCallbacks {
  Preprocessor &PP;
  IndexingContext &IndexCtx;

public:
  IndexPPCallbacks(Preprocessor &PP, IndexingContext &indexCtx)
    : PP(PP), IndexCtx(indexCtx) { }

  virtual void InclusionDirective(SourceLocation HashLoc,
                                  const Token &IncludeTok,
                                  StringRef FileName,
                                  bool IsAngled,
                                  const FileEntry *File,
                                  SourceLocation EndLoc,
                                  StringRef SearchPath,
                                  StringRef RelativePath) {
    bool isImport = (IncludeTok.is(tok::identifier) &&
            IncludeTok.getIdentifierInfo()->getPPKeywordID() == tok::pp_import);
    IndexCtx.ppIncludedFile(HashLoc, FileName, File, isImport, IsAngled);
  }

  /// MacroDefined - This hook is called whenever a macro definition is seen.
  virtual void MacroDefined(const Token &Id, const MacroInfo *MI) {
    if (MI->isBuiltinMacro())
      return;
    if (IndexCtx.isNotFromSourceFile(MI->getDefinitionLoc()))
      return;

    SourceLocation Loc = MI->getDefinitionLoc();
    SourceLocation DefBegin = MI->tokens_empty() ? Loc
                                     : MI->getReplacementToken(0).getLocation();
    IndexCtx.ppMacroDefined(Loc,
                            Id.getIdentifierInfo()->getName(),
                            DefBegin,
                            MI->getDefinitionLength(PP.getSourceManager()),
                            MI);
  }

  /// MacroUndefined - This hook is called whenever a macro #undef is seen.
  /// MI is released immediately following this callback.
  virtual void MacroUndefined(const Token &MacroNameTok, const MacroInfo *MI) {
    if (MI->isBuiltinMacro())
      return;
    if (IndexCtx.isNotFromSourceFile(MI->getDefinitionLoc()))
      return;

    SourceLocation Loc = MacroNameTok.getLocation();
    IndexCtx.ppMacroUndefined(Loc,
                            MacroNameTok.getIdentifierInfo()->getName(),
                            MI);
  }

  /// MacroExpands - This is called by when a macro invocation is found.
  virtual void MacroExpands(const Token &MacroNameTok, const MacroInfo* MI,
                            SourceRange Range) {
    if (MI->isBuiltinMacro())
      return;
    if (IndexCtx.isNotFromSourceFile(MI->getDefinitionLoc()))
      return;

    SourceLocation Loc = MacroNameTok.getLocation();
    IndexCtx.ppMacroExpanded(Loc,
                             MacroNameTok.getIdentifierInfo()->getName(),
                             MI);
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

public:
  explicit IndexingConsumer(IndexingContext &indexCtx)
    : IndexCtx(indexCtx) { }

  // ASTConsumer Implementation

  virtual void Initialize(ASTContext &Context) {
    IndexCtx.setASTContext(Context);
    IndexCtx.invokeStartedTranslationUnit();
  }

  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    IndexCtx.invokeFinishedTranslationUnit();
  }

  virtual void HandleTopLevelDecl(DeclGroupRef DG) {
    IndexCtx.indexDeclGroupRef(DG);
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
};

//===----------------------------------------------------------------------===//
// IndexingDiagnosticConsumer
//===----------------------------------------------------------------------===//

class IndexingDiagnosticConsumer : public DiagnosticConsumer {
  IndexingContext &IndexCtx;
  
public:
  explicit IndexingDiagnosticConsumer(IndexingContext &indexCtx)
    : IndexCtx(indexCtx) {}
  
  virtual void HandleDiagnostic(DiagnosticsEngine::Level Level,
                                const Diagnostic &Info) {
    // Default implementation (Warnings/errors count).
    DiagnosticConsumer::HandleDiagnostic(Level, Info);

    IndexCtx.handleDiagnostic(StoredDiagnostic(Level, Info));
  }

  DiagnosticConsumer *clone(DiagnosticsEngine &Diags) const {
    return new IgnoringDiagConsumer();
  }
};

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

public:
  IndexingFrontendAction(CXClientData clientData,
                         IndexerCallbacks &indexCallbacks,
                         unsigned indexOptions,
                         CXTranslationUnit cxTU)
    : IndexCtx(clientData, indexCallbacks, indexOptions, cxTU) { }

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         StringRef InFile) {
    CI.getDiagnostics().setClient(new IndexingDiagnosticConsumer(IndexCtx),
                                  /*own=*/true);
    IndexCtx.setASTContext(CI.getASTContext());
    Preprocessor &PP = CI.getPreprocessor();
    PP.addPPCallbacks(new IndexPPCallbacks(PP, IndexCtx));
    return new IndexingConsumer(IndexCtx);
  }

  virtual TranslationUnitKind getTranslationUnitKind() { return TU_Prefix; }
  virtual bool hasCodeCompletionSupport() const { return false; }
};

//===----------------------------------------------------------------------===//
// clang_indexTranslationUnit Implementation
//===----------------------------------------------------------------------===//

struct IndexTranslationUnitInfo {
  CXIndex CIdx;
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

static void clang_indexTranslationUnit_Impl(void *UserData) {
  IndexTranslationUnitInfo *ITUI =
    static_cast<IndexTranslationUnitInfo*>(UserData);
  CXIndex CIdx = ITUI->CIdx;
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

  if (!CIdx)
    return;
  if (!client_index_callbacks || index_callbacks_size == 0)
    return;

  IndexerCallbacks CB;
  memset(&CB, 0, sizeof(CB));
  unsigned ClientCBSize = index_callbacks_size < sizeof(CB)
                                  ? index_callbacks_size : sizeof(CB);
  memcpy(&CB, client_index_callbacks, ClientCBSize);

  CIndexer *CXXIdx = static_cast<CIndexer *>(CIdx);

  (void)CXXIdx;
  (void)TU_options;
  
  CaptureDiagnosticConsumer *CaptureDiag = new CaptureDiagnosticConsumer();

  // Configure the diagnostics.
  DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine>
    Diags(CompilerInstance::createDiagnostics(DiagOpts, num_command_line_args, 
                                                command_line_args,
                                                CaptureDiag,
                                                /*ShouldOwnClient=*/true));

  // Recover resources if we crash before exiting this function.
  llvm::CrashRecoveryContextCleanupRegistrar<DiagnosticsEngine,
    llvm::CrashRecoveryContextReleaseRefCleanup<DiagnosticsEngine> >
    DiagCleanup(Diags.getPtr());
  
  llvm::OwningPtr<std::vector<const char *> > 
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
  
  llvm::IntrusiveRefCntPtr<CompilerInvocation>
    CInvok(createInvocationFromCommandLine(*Args, Diags));

  if (!CInvok)
    return;

  // Recover resources if we crash before exiting this function.
  llvm::CrashRecoveryContextCleanupRegistrar<CompilerInvocation,
    llvm::CrashRecoveryContextReleaseRefCleanup<CompilerInvocation> >
    CInvokCleanup(CInvok.getPtr());

  if (CInvok->getFrontendOpts().Inputs.empty())
    return;

  llvm::OwningPtr<MemBufferOwner> BufOwner(new MemBufferOwner());

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
  CInvok->getLangOpts().SpellChecking = false;

  if (!requestedToGetTU)
    CInvok->getPreprocessorOpts().DetailedRecord = false;

  ASTUnit *Unit = ASTUnit::create(CInvok.getPtr(), Diags);
  llvm::OwningPtr<CXTUOwner> CXTU(new CXTUOwner(MakeCXTranslationUnit(Unit)));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<CXTUOwner>
    CXTUCleanup(CXTU.get());

  llvm::OwningPtr<IndexingFrontendAction> IndexAction;
  IndexAction.reset(new IndexingFrontendAction(client_data, CB,
                                               index_options, CXTU->getTU()));

  // Recover resources if we crash before exiting this method.
  llvm::CrashRecoveryContextCleanupRegistrar<IndexingFrontendAction>
    IndexActionCleanup(IndexAction.get());

  Unit = ASTUnit::LoadFromCompilerInvocationAction(CInvok.getPtr(), Diags,
                                                       IndexAction.get(),
                                                       Unit);
  if (!Unit)
    return;

  // FIXME: Set state of the ASTUnit according to the TU_options.
  if (out_TU)
    *out_TU = CXTU->takeTU();

  ITUI->result = 0; // success.
}

//===----------------------------------------------------------------------===//
// libclang public APIs.
//===----------------------------------------------------------------------===//

extern "C" {

int clang_indexTranslationUnit(CXIndex CIdx,
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
  IndexTranslationUnitInfo ITUI = { CIdx, client_data, index_callbacks,
                                    index_callbacks_size, index_options,
                                    source_filename, command_line_args,
                                    num_command_line_args, unsaved_files,
                                    num_unsaved_files, out_TU, TU_options, 0 };

  if (getenv("CINDEXTEST_NOTHREADS")) {
    clang_indexTranslationUnit_Impl(&ITUI);
    return ITUI.result;
  }

  llvm::CrashRecoveryContext CRC;

  if (!RunSafely(CRC, clang_indexTranslationUnit_Impl, &ITUI)) {
    fprintf(stderr, "libclang: crash detected during parsing: {\n");
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

void clang_indexLoc_getFileLocation(CXIdxLoc location,
                                    CXIdxFile *indexFile,
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

