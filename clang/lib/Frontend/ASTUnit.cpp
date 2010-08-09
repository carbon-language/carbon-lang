//===--- ASTUnit.cpp - ASTUnit utility ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ASTUnit Implementation.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/PCHWriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/PCHReader.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
using namespace clang;

/// \brief After failing to build a precompiled preamble (due to
/// errors in the source that occurs in the preamble), the number of
/// reparses during which we'll skip even trying to precompile the
/// preamble.
const unsigned DefaultPreambleRebuildInterval = 5;

ASTUnit::ASTUnit(bool _MainFileIsAST)
  : CaptureDiagnostics(false), MainFileIsAST(_MainFileIsAST), 
    CompleteTranslationUnit(true), ConcurrencyCheckValue(CheckUnlocked), 
    PreambleRebuildCounter(0), SavedMainFileBuffer(0) { 
}

ASTUnit::~ASTUnit() {
  ConcurrencyCheckValue = CheckLocked;
  CleanTemporaryFiles();
  if (!PreambleFile.empty())
    llvm::sys::Path(PreambleFile).eraseFromDisk();
  
  // Free the buffers associated with remapped files. We are required to
  // perform this operation here because we explicitly request that the
  // compiler instance *not* free these buffers for each invocation of the
  // parser.
  if (Invocation.get()) {
    PreprocessorOptions &PPOpts = Invocation->getPreprocessorOpts();
    for (PreprocessorOptions::remapped_file_buffer_iterator
           FB = PPOpts.remapped_file_buffer_begin(),
           FBEnd = PPOpts.remapped_file_buffer_end();
         FB != FBEnd;
         ++FB)
      delete FB->second;
  }
  
  delete SavedMainFileBuffer;
  
  for (unsigned I = 0, N = Timers.size(); I != N; ++I)
    delete Timers[I];
}

void ASTUnit::CleanTemporaryFiles() {
  for (unsigned I = 0, N = TemporaryFiles.size(); I != N; ++I)
    TemporaryFiles[I].eraseFromDisk();
  TemporaryFiles.clear();
}

namespace {

/// \brief Gathers information from PCHReader that will be used to initialize
/// a Preprocessor.
class PCHInfoCollector : public PCHReaderListener {
  LangOptions &LangOpt;
  HeaderSearch &HSI;
  std::string &TargetTriple;
  std::string &Predefines;
  unsigned &Counter;

  unsigned NumHeaderInfos;

public:
  PCHInfoCollector(LangOptions &LangOpt, HeaderSearch &HSI,
                   std::string &TargetTriple, std::string &Predefines,
                   unsigned &Counter)
    : LangOpt(LangOpt), HSI(HSI), TargetTriple(TargetTriple),
      Predefines(Predefines), Counter(Counter), NumHeaderInfos(0) {}

  virtual bool ReadLanguageOptions(const LangOptions &LangOpts) {
    LangOpt = LangOpts;
    return false;
  }

  virtual bool ReadTargetTriple(llvm::StringRef Triple) {
    TargetTriple = Triple;
    return false;
  }

  virtual bool ReadPredefinesBuffer(const PCHPredefinesBlocks &Buffers,
                                    llvm::StringRef OriginalFileName,
                                    std::string &SuggestedPredefines) {
    Predefines = Buffers[0].Data;
    for (unsigned I = 1, N = Buffers.size(); I != N; ++I) {
      Predefines += Buffers[I].Data;
    }
    return false;
  }

  virtual void ReadHeaderFileInfo(const HeaderFileInfo &HFI, unsigned ID) {
    HSI.setHeaderFileInfoForUID(HFI, NumHeaderInfos++);
  }

  virtual void ReadCounter(unsigned Value) {
    Counter = Value;
  }
};

class StoredDiagnosticClient : public DiagnosticClient {
  llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiags;
  
public:
  explicit StoredDiagnosticClient(
                          llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiags)
    : StoredDiags(StoredDiags) { }
  
  virtual void HandleDiagnostic(Diagnostic::Level Level,
                                const DiagnosticInfo &Info);
};

/// \brief RAII object that optionally captures diagnostics, if
/// there is no diagnostic client to capture them already.
class CaptureDroppedDiagnostics {
  Diagnostic &Diags;
  StoredDiagnosticClient Client;
  DiagnosticClient *PreviousClient;

public:
  CaptureDroppedDiagnostics(bool RequestCapture, Diagnostic &Diags, 
                           llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiags)
    : Diags(Diags), Client(StoredDiags), PreviousClient(Diags.getClient()) 
  {
    if (RequestCapture || Diags.getClient() == 0)
      Diags.setClient(&Client);
  }

  ~CaptureDroppedDiagnostics() {
    Diags.setClient(PreviousClient);
  }
};

} // anonymous namespace

void StoredDiagnosticClient::HandleDiagnostic(Diagnostic::Level Level,
                                              const DiagnosticInfo &Info) {
  StoredDiags.push_back(StoredDiagnostic(Level, Info));
}

const std::string &ASTUnit::getOriginalSourceFileName() {
  return OriginalSourceFile;
}

const std::string &ASTUnit::getPCHFileName() {
  assert(isMainFileAST() && "Not an ASTUnit from a PCH file!");
  return static_cast<PCHReader *>(Ctx->getExternalSource())->getFileName();
}

ASTUnit *ASTUnit::LoadFromPCHFile(const std::string &Filename,
                                  llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                  bool OnlyLocalDecls,
                                  RemappedFile *RemappedFiles,
                                  unsigned NumRemappedFiles,
                                  bool CaptureDiagnostics) {
  llvm::OwningPtr<ASTUnit> AST(new ASTUnit(true));
  
  if (!Diags.getPtr()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    DiagnosticOptions DiagOpts;
    Diags = CompilerInstance::createDiagnostics(DiagOpts, 0, 0);
  }

  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->Diagnostics = Diags;
  AST->FileMgr.reset(new FileManager);
  AST->SourceMgr.reset(new SourceManager(AST->getDiagnostics()));
  AST->HeaderInfo.reset(new HeaderSearch(AST->getFileManager()));

  // If requested, capture diagnostics in the ASTUnit.
  CaptureDroppedDiagnostics Capture(CaptureDiagnostics, AST->getDiagnostics(),
                                    AST->StoredDiagnostics);

  for (unsigned I = 0; I != NumRemappedFiles; ++I) {
    // Create the file entry for the file that we're mapping from.
    const FileEntry *FromFile
      = AST->getFileManager().getVirtualFile(RemappedFiles[I].first,
                                    RemappedFiles[I].second->getBufferSize(),
                                             0);
    if (!FromFile) {
      AST->getDiagnostics().Report(diag::err_fe_remap_missing_from_file)
        << RemappedFiles[I].first;
      delete RemappedFiles[I].second;
      continue;
    }
    
    // Override the contents of the "from" file with the contents of
    // the "to" file.
    AST->getSourceManager().overrideFileContents(FromFile, 
                                                 RemappedFiles[I].second);    
  }
  
  // Gather Info for preprocessor construction later on.

  LangOptions LangInfo;
  HeaderSearch &HeaderInfo = *AST->HeaderInfo.get();
  std::string TargetTriple;
  std::string Predefines;
  unsigned Counter;

  llvm::OwningPtr<PCHReader> Reader;
  llvm::OwningPtr<ExternalASTSource> Source;

  Reader.reset(new PCHReader(AST->getSourceManager(), AST->getFileManager(),
                             AST->getDiagnostics()));
  Reader->setListener(new PCHInfoCollector(LangInfo, HeaderInfo, TargetTriple,
                                           Predefines, Counter));

  switch (Reader->ReadPCH(Filename)) {
  case PCHReader::Success:
    break;

  case PCHReader::Failure:
  case PCHReader::IgnorePCH:
    AST->getDiagnostics().Report(diag::err_fe_unable_to_load_pch);
    return NULL;
  }

  AST->OriginalSourceFile = Reader->getOriginalSourceFile();

  // PCH loaded successfully. Now create the preprocessor.

  // Get information about the target being compiled for.
  //
  // FIXME: This is broken, we should store the TargetOptions in the PCH.
  TargetOptions TargetOpts;
  TargetOpts.ABI = "";
  TargetOpts.CXXABI = "itanium";
  TargetOpts.CPU = "";
  TargetOpts.Features.clear();
  TargetOpts.Triple = TargetTriple;
  AST->Target.reset(TargetInfo::CreateTargetInfo(AST->getDiagnostics(),
                                                 TargetOpts));
  AST->PP.reset(new Preprocessor(AST->getDiagnostics(), LangInfo, 
                                 *AST->Target.get(),
                                 AST->getSourceManager(), HeaderInfo));
  Preprocessor &PP = *AST->PP.get();

  PP.setPredefines(Reader->getSuggestedPredefines());
  PP.setCounterValue(Counter);
  Reader->setPreprocessor(PP);

  // Create and initialize the ASTContext.

  AST->Ctx.reset(new ASTContext(LangInfo,
                                AST->getSourceManager(),
                                *AST->Target.get(),
                                PP.getIdentifierTable(),
                                PP.getSelectorTable(),
                                PP.getBuiltinInfo(),
                                /* size_reserve = */0));
  ASTContext &Context = *AST->Ctx.get();

  Reader->InitializeContext(Context);

  // Attach the PCH reader to the AST context as an external AST
  // source, so that declarations will be deserialized from the
  // PCH file as needed.
  Source.reset(Reader.take());
  Context.setExternalSource(Source);

  return AST.take();
}

namespace {

class TopLevelDeclTrackerConsumer : public ASTConsumer {
  ASTUnit &Unit;

public:
  TopLevelDeclTrackerConsumer(ASTUnit &_Unit) : Unit(_Unit) {}

  void HandleTopLevelDecl(DeclGroupRef D) {
    for (DeclGroupRef::iterator it = D.begin(), ie = D.end(); it != ie; ++it) {
      Decl *D = *it;
      // FIXME: Currently ObjC method declarations are incorrectly being
      // reported as top-level declarations, even though their DeclContext
      // is the containing ObjC @interface/@implementation.  This is a
      // fundamental problem in the parser right now.
      if (isa<ObjCMethodDecl>(D))
        continue;
      Unit.addTopLevelDecl(D);
    }
  }

  // We're not interested in "interesting" decls.
  void HandleInterestingDecl(DeclGroupRef) {}
};

class TopLevelDeclTrackerAction : public ASTFrontendAction {
public:
  ASTUnit &Unit;

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile) {
    return new TopLevelDeclTrackerConsumer(Unit);
  }

public:
  TopLevelDeclTrackerAction(ASTUnit &_Unit) : Unit(_Unit) {}

  virtual bool hasCodeCompletionSupport() const { return false; }
  virtual bool usesCompleteTranslationUnit()  { 
    return Unit.isCompleteTranslationUnit(); 
  }
};

class PrecompilePreambleConsumer : public PCHGenerator {
  ASTUnit &Unit;
  std::vector<Decl *> TopLevelDecls;

public:
  PrecompilePreambleConsumer(ASTUnit &Unit,
                             const Preprocessor &PP, bool Chaining,
                             const char *isysroot, llvm::raw_ostream *Out)
    : PCHGenerator(PP, Chaining, isysroot, Out), Unit(Unit) { }

  virtual void HandleTopLevelDecl(DeclGroupRef D) {
    for (DeclGroupRef::iterator it = D.begin(), ie = D.end(); it != ie; ++it) {
      Decl *D = *it;
      // FIXME: Currently ObjC method declarations are incorrectly being
      // reported as top-level declarations, even though their DeclContext
      // is the containing ObjC @interface/@implementation.  This is a
      // fundamental problem in the parser right now.
      if (isa<ObjCMethodDecl>(D))
        continue;
      TopLevelDecls.push_back(D);
    }
  }

  virtual void HandleTranslationUnit(ASTContext &Ctx) {
    PCHGenerator::HandleTranslationUnit(Ctx);
    if (!Unit.getDiagnostics().hasErrorOccurred()) {
      // Translate the top-level declarations we captured during
      // parsing into declaration IDs in the precompiled
      // preamble. This will allow us to deserialize those top-level
      // declarations when requested.
      for (unsigned I = 0, N = TopLevelDecls.size(); I != N; ++I)
        Unit.addTopLevelDeclFromPreamble(
                                      getWriter().getDeclID(TopLevelDecls[I]));
    }
  }
};

class PrecompilePreambleAction : public ASTFrontendAction {
  ASTUnit &Unit;

public:
  explicit PrecompilePreambleAction(ASTUnit &Unit) : Unit(Unit) {}

  virtual ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                         llvm::StringRef InFile) {
    std::string Sysroot;
    llvm::raw_ostream *OS = 0;
    bool Chaining;
    if (GeneratePCHAction::ComputeASTConsumerArguments(CI, InFile, Sysroot, 
                                                       OS, Chaining))
      return 0;
    
    const char *isysroot = CI.getFrontendOpts().RelocatablePCH ?
                             Sysroot.c_str() : 0;  
    return new PrecompilePreambleConsumer(Unit, CI.getPreprocessor(), Chaining,
                                          isysroot, OS);
  }

  virtual bool hasCodeCompletionSupport() const { return false; }
  virtual bool hasASTFileSupport() const { return false; }
  virtual bool usesCompleteTranslationUnit() { return false; }
};

}

/// Parse the source file into a translation unit using the given compiler
/// invocation, replacing the current translation unit.
///
/// \returns True if a failure occurred that causes the ASTUnit not to
/// contain any translation-unit information, false otherwise.
bool ASTUnit::Parse(llvm::MemoryBuffer *OverrideMainBuffer) {
  delete SavedMainFileBuffer;
  SavedMainFileBuffer = 0;
  
  if (!Invocation.get())
    return true;
  
  // Create the compiler instance to use for building the AST.
  CompilerInstance Clang;
  Clang.setInvocation(Invocation.take());
  OriginalSourceFile = Clang.getFrontendOpts().Inputs[0].second;
    
  // Set up diagnostics, capturing any diagnostics that would
  // otherwise be dropped.
  Clang.setDiagnostics(&getDiagnostics());
  CaptureDroppedDiagnostics Capture(CaptureDiagnostics, 
                                    getDiagnostics(),
                                    StoredDiagnostics);
  Clang.setDiagnosticClient(getDiagnostics().getClient());
  
  // Create the target instance.
  Clang.setTarget(TargetInfo::CreateTargetInfo(Clang.getDiagnostics(),
                                               Clang.getTargetOpts()));
  if (!Clang.hasTarget()) {
    Clang.takeDiagnosticClient();
    return true;
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang.getTarget().setForcedLangOptions(Clang.getLangOpts());
  
  assert(Clang.getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_LLVM_IR &&
         "IR inputs not support here!");

  // Configure the various subsystems.
  // FIXME: Should we retain the previous file manager?
  FileMgr.reset(new FileManager);
  SourceMgr.reset(new SourceManager(getDiagnostics()));
  Ctx.reset();
  PP.reset();
  
  // Clear out old caches and data.
  TopLevelDecls.clear();
  CleanTemporaryFiles();
  PreprocessedEntitiesByFile.clear();

  if (!OverrideMainBuffer)
    StoredDiagnostics.clear();
    
  // Create a file manager object to provide access to and cache the filesystem.
  Clang.setFileManager(&getFileManager());
  
  // Create the source manager.
  Clang.setSourceManager(&getSourceManager());
  
  // If the main file has been overridden due to the use of a preamble,
  // make that override happen and introduce the preamble.
  PreprocessorOptions &PreprocessorOpts = Clang.getPreprocessorOpts();
  std::string PriorImplicitPCHInclude;
  if (OverrideMainBuffer) {
    PreprocessorOpts.addRemappedFile(OriginalSourceFile, OverrideMainBuffer);
    PreprocessorOpts.PrecompiledPreambleBytes.first = Preamble.size();
    PreprocessorOpts.PrecompiledPreambleBytes.second
                                                    = PreambleEndsAtStartOfLine;
    PriorImplicitPCHInclude = PreprocessorOpts.ImplicitPCHInclude;
    PreprocessorOpts.ImplicitPCHInclude = PreambleFile;
    PreprocessorOpts.DisablePCHValidation = true;
    
    // Keep track of the override buffer;
    SavedMainFileBuffer = OverrideMainBuffer;

    // The stored diagnostic has the old source manager in it; update
    // the locations to refer into the new source manager. Since we've
    // been careful to make sure that the source manager's state
    // before and after are identical, so that we can reuse the source
    // location itself.
    for (unsigned I = 0, N = StoredDiagnostics.size(); I != N; ++I) {
      FullSourceLoc Loc(StoredDiagnostics[I].getLocation(),
                        getSourceManager());
      StoredDiagnostics[I].setLocation(Loc);
    }
  }
  
  llvm::OwningPtr<TopLevelDeclTrackerAction> Act;
  Act.reset(new TopLevelDeclTrackerAction(*this));
  if (!Act->BeginSourceFile(Clang, Clang.getFrontendOpts().Inputs[0].second,
                            Clang.getFrontendOpts().Inputs[0].first))
    goto error;
  
  Act->Execute();
  
  // Steal the created target, context, and preprocessor, and take back the
  // source and file managers.
  Ctx.reset(Clang.takeASTContext());
  PP.reset(Clang.takePreprocessor());
  Clang.takeSourceManager();
  Clang.takeFileManager();
  Target.reset(Clang.takeTarget());
  
  Act->EndSourceFile();

  // Remove the overridden buffer we used for the preamble.
  if (OverrideMainBuffer) {
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    PreprocessorOpts.ImplicitPCHInclude = PriorImplicitPCHInclude;
  }

  Clang.takeDiagnosticClient();
  
  Invocation.reset(Clang.takeInvocation());
  return false;
  
error:
  // Remove the overridden buffer we used for the preamble.
  if (OverrideMainBuffer) {
    PreprocessorOpts.eraseRemappedFile(
                               PreprocessorOpts.remapped_file_buffer_end() - 1);
    PreprocessorOpts.DisablePCHValidation = true;
    PreprocessorOpts.ImplicitPCHInclude = PriorImplicitPCHInclude;
  }
  
  Clang.takeSourceManager();
  Clang.takeFileManager();
  Clang.takeDiagnosticClient();
  Invocation.reset(Clang.takeInvocation());
  return true;
}

/// \brief Simple function to retrieve a path for a preamble precompiled header.
static std::string GetPreamblePCHPath() {
  // FIXME: This is lame; sys::Path should provide this function (in particular,
  // it should know how to find the temporary files dir).
  // FIXME: This is really lame. I copied this code from the Driver!
  std::string Error;
  const char *TmpDir = ::getenv("TMPDIR");
  if (!TmpDir)
    TmpDir = ::getenv("TEMP");
  if (!TmpDir)
    TmpDir = ::getenv("TMP");
  if (!TmpDir)
    TmpDir = "/tmp";
  llvm::sys::Path P(TmpDir);
  P.appendComponent("preamble");
  if (P.createTemporaryFileOnDisk())
    return std::string();
  
  P.appendSuffix("pch");
  return P.str();
}

/// \brief Compute the preamble for the main file, providing the source buffer
/// that corresponds to the main file along with a pair (bytes, start-of-line)
/// that describes the preamble.
std::pair<llvm::MemoryBuffer *, std::pair<unsigned, bool> > 
ASTUnit::ComputePreamble(CompilerInvocation &Invocation, 
                         unsigned MaxLines, bool &CreatedBuffer) {
  FrontendOptions &FrontendOpts = Invocation.getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts
    = Invocation.getPreprocessorOpts();
  CreatedBuffer = false;
  
  // Try to determine if the main file has been remapped, either from the 
  // command line (to another file) or directly through the compiler invocation
  // (to a memory buffer).
  llvm::MemoryBuffer *Buffer = 0;
  llvm::sys::PathWithStatus MainFilePath(FrontendOpts.Inputs[0].second);
  if (const llvm::sys::FileStatus *MainFileStatus = MainFilePath.getFileStatus()) {
    // Check whether there is a file-file remapping of the main file
    for (PreprocessorOptions::remapped_file_iterator
          M = PreprocessorOpts.remapped_file_begin(),
          E = PreprocessorOpts.remapped_file_end();
         M != E;
         ++M) {
      llvm::sys::PathWithStatus MPath(M->first);    
      if (const llvm::sys::FileStatus *MStatus = MPath.getFileStatus()) {
        if (MainFileStatus->uniqueID == MStatus->uniqueID) {
          // We found a remapping. Try to load the resulting, remapped source.
          if (CreatedBuffer) {
            delete Buffer;
            CreatedBuffer = false;
          }
          
          Buffer = llvm::MemoryBuffer::getFile(M->second);
          if (!Buffer)
            return std::make_pair((llvm::MemoryBuffer*)0, 
                                  std::make_pair(0, true));
          CreatedBuffer = true;
          
          // Remove this remapping. We've captured the buffer already.
          M = PreprocessorOpts.eraseRemappedFile(M);
          E = PreprocessorOpts.remapped_file_end();
        }
      }
    }
    
    // Check whether there is a file-buffer remapping. It supercedes the
    // file-file remapping.
    for (PreprocessorOptions::remapped_file_buffer_iterator
           M = PreprocessorOpts.remapped_file_buffer_begin(),
           E = PreprocessorOpts.remapped_file_buffer_end();
         M != E;
         ++M) {
      llvm::sys::PathWithStatus MPath(M->first);    
      if (const llvm::sys::FileStatus *MStatus = MPath.getFileStatus()) {
        if (MainFileStatus->uniqueID == MStatus->uniqueID) {
          // We found a remapping. 
          if (CreatedBuffer) {
            delete Buffer;
            CreatedBuffer = false;
          }
          
          Buffer = const_cast<llvm::MemoryBuffer *>(M->second);

          // Remove this remapping. We've captured the buffer already.
          M = PreprocessorOpts.eraseRemappedFile(M);
          E = PreprocessorOpts.remapped_file_buffer_end();
        }
      }
    }
  }
  
  // If the main source file was not remapped, load it now.
  if (!Buffer) {
    Buffer = llvm::MemoryBuffer::getFile(FrontendOpts.Inputs[0].second);
    if (!Buffer)
      return std::make_pair((llvm::MemoryBuffer*)0, std::make_pair(0, true));    
    
    CreatedBuffer = true;
  }
  
  return std::make_pair(Buffer, Lexer::ComputePreamble(Buffer, MaxLines));
}

static llvm::MemoryBuffer *CreatePaddedMainFileBuffer(llvm::MemoryBuffer *Old,
                                                      bool DeleteOld,
                                                      unsigned NewSize,
                                                      llvm::StringRef NewName) {
  llvm::MemoryBuffer *Result
    = llvm::MemoryBuffer::getNewUninitMemBuffer(NewSize, NewName);
  memcpy(const_cast<char*>(Result->getBufferStart()), 
         Old->getBufferStart(), Old->getBufferSize());
  memset(const_cast<char*>(Result->getBufferStart()) + Old->getBufferSize(), 
         ' ', NewSize - Old->getBufferSize() - 1);
  const_cast<char*>(Result->getBufferEnd())[-1] = '\n';  
  
  if (DeleteOld)
    delete Old;
  
  return Result;
}

/// \brief Attempt to build or re-use a precompiled preamble when (re-)parsing
/// the source file.
///
/// This routine will compute the preamble of the main source file. If a
/// non-trivial preamble is found, it will precompile that preamble into a 
/// precompiled header so that the precompiled preamble can be used to reduce
/// reparsing time. If a precompiled preamble has already been constructed,
/// this routine will determine if it is still valid and, if so, avoid 
/// rebuilding the precompiled preamble.
///
/// \param AllowRebuild When true (the default), this routine is
/// allowed to rebuild the precompiled preamble if it is found to be
/// out-of-date.
///
/// \param MaxLines When non-zero, the maximum number of lines that
/// can occur within the preamble.
///
/// \returns If the precompiled preamble can be used, returns a newly-allocated
/// buffer that should be used in place of the main file when doing so.
/// Otherwise, returns a NULL pointer.
llvm::MemoryBuffer *ASTUnit::getMainBufferWithPrecompiledPreamble(
                                                           bool AllowRebuild,
                                                           unsigned MaxLines) {
  CompilerInvocation PreambleInvocation(*Invocation);
  FrontendOptions &FrontendOpts = PreambleInvocation.getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts
    = PreambleInvocation.getPreprocessorOpts();

  bool CreatedPreambleBuffer = false;
  std::pair<llvm::MemoryBuffer *, std::pair<unsigned, bool> > NewPreamble 
    = ComputePreamble(PreambleInvocation, MaxLines, CreatedPreambleBuffer);

  if (!NewPreamble.second.first) {
    // We couldn't find a preamble in the main source. Clear out the current
    // preamble, if we have one. It's obviously no good any more.
    Preamble.clear();
    if (!PreambleFile.empty()) {
      llvm::sys::Path(PreambleFile).eraseFromDisk();
      PreambleFile.clear();
    }
    if (CreatedPreambleBuffer)
      delete NewPreamble.first;

    // The next time we actually see a preamble, precompile it.
    PreambleRebuildCounter = 1;
    return 0;
  }
  
  if (!Preamble.empty()) {
    // We've previously computed a preamble. Check whether we have the same
    // preamble now that we did before, and that there's enough space in
    // the main-file buffer within the precompiled preamble to fit the
    // new main file.
    if (Preamble.size() == NewPreamble.second.first &&
        PreambleEndsAtStartOfLine == NewPreamble.second.second &&
        NewPreamble.first->getBufferSize() < PreambleReservedSize-2 &&
        memcmp(&Preamble[0], NewPreamble.first->getBufferStart(),
               NewPreamble.second.first) == 0) {
      // The preamble has not changed. We may be able to re-use the precompiled
      // preamble.

      // Check that none of the files used by the preamble have changed.
      bool AnyFileChanged = false;
          
      // First, make a record of those files that have been overridden via
      // remapping or unsaved_files.
      llvm::StringMap<std::pair<off_t, time_t> > OverriddenFiles;
      for (PreprocessorOptions::remapped_file_iterator
                R = PreprocessorOpts.remapped_file_begin(),
             REnd = PreprocessorOpts.remapped_file_end();
           !AnyFileChanged && R != REnd;
           ++R) {
        struct stat StatBuf;
        if (stat(R->second.c_str(), &StatBuf)) {
          // If we can't stat the file we're remapping to, assume that something
          // horrible happened.
          AnyFileChanged = true;
          break;
        }
        
        OverriddenFiles[R->first] = std::make_pair(StatBuf.st_size, 
                                                   StatBuf.st_mtime);
      }
      for (PreprocessorOptions::remapped_file_buffer_iterator
                R = PreprocessorOpts.remapped_file_buffer_begin(),
             REnd = PreprocessorOpts.remapped_file_buffer_end();
           !AnyFileChanged && R != REnd;
           ++R) {
        // FIXME: Should we actually compare the contents of file->buffer
        // remappings?
        OverriddenFiles[R->first] = std::make_pair(R->second->getBufferSize(), 
                                                   0);
      }
       
      // Check whether anything has changed.
      for (llvm::StringMap<std::pair<off_t, time_t> >::iterator 
             F = FilesInPreamble.begin(), FEnd = FilesInPreamble.end();
           !AnyFileChanged && F != FEnd; 
           ++F) {
        llvm::StringMap<std::pair<off_t, time_t> >::iterator Overridden
          = OverriddenFiles.find(F->first());
        if (Overridden != OverriddenFiles.end()) {
          // This file was remapped; check whether the newly-mapped file 
          // matches up with the previous mapping.
          if (Overridden->second != F->second)
            AnyFileChanged = true;
          continue;
        }
        
        // The file was not remapped; check whether it has changed on disk.
        struct stat StatBuf;
        if (stat(F->first(), &StatBuf)) {
          // If we can't stat the file, assume that something horrible happened.
          AnyFileChanged = true;
        } else if (StatBuf.st_size != F->second.first || 
                   StatBuf.st_mtime != F->second.second)
          AnyFileChanged = true;
      }
          
      if (!AnyFileChanged) {
        // Okay! We can re-use the precompiled preamble.

        // Set the state of the diagnostic object to mimic its state
        // after parsing the preamble.
        getDiagnostics().Reset();
        getDiagnostics().setNumWarnings(NumWarningsInPreamble);
        if (StoredDiagnostics.size() > NumStoredDiagnosticsInPreamble)
          StoredDiagnostics.erase(
            StoredDiagnostics.begin() + NumStoredDiagnosticsInPreamble,
                                  StoredDiagnostics.end());

        // Create a version of the main file buffer that is padded to
        // buffer size we reserved when creating the preamble.
        return CreatePaddedMainFileBuffer(NewPreamble.first, 
                                          CreatedPreambleBuffer,
                                          PreambleReservedSize,
                                          FrontendOpts.Inputs[0].second);
      }
    }

    // If we aren't allowed to rebuild the precompiled preamble, just
    // return now.
    if (!AllowRebuild)
      return 0;
    
    // We can't reuse the previously-computed preamble. Build a new one.
    Preamble.clear();
    llvm::sys::Path(PreambleFile).eraseFromDisk();
    PreambleRebuildCounter = 1;
  } else if (!AllowRebuild) {
    // We aren't allowed to rebuild the precompiled preamble; just
    // return now.
    return 0;
  }

  // If the preamble rebuild counter > 1, it's because we previously
  // failed to build a preamble and we're not yet ready to try
  // again. Decrement the counter and return a failure.
  if (PreambleRebuildCounter > 1) {
    --PreambleRebuildCounter;
    return 0;
  }

  // We did not previously compute a preamble, or it can't be reused anyway.
  llvm::Timer *PreambleTimer = 0;
  if (TimerGroup.get()) {
    PreambleTimer = new llvm::Timer("Precompiling preamble", *TimerGroup);
    PreambleTimer->startTimer();
    Timers.push_back(PreambleTimer);
  }
  
  // Create a new buffer that stores the preamble. The buffer also contains
  // extra space for the original contents of the file (which will be present
  // when we actually parse the file) along with more room in case the file
  // grows.  
  PreambleReservedSize = NewPreamble.first->getBufferSize();
  if (PreambleReservedSize < 4096)
    PreambleReservedSize = 8191;
  else
    PreambleReservedSize *= 2;

  // Save the preamble text for later; we'll need to compare against it for
  // subsequent reparses.
  Preamble.assign(NewPreamble.first->getBufferStart(), 
                  NewPreamble.first->getBufferStart() 
                                                  + NewPreamble.second.first);
  PreambleEndsAtStartOfLine = NewPreamble.second.second;

  llvm::MemoryBuffer *PreambleBuffer
    = llvm::MemoryBuffer::getNewUninitMemBuffer(PreambleReservedSize,
                                                FrontendOpts.Inputs[0].second);
  memcpy(const_cast<char*>(PreambleBuffer->getBufferStart()), 
         NewPreamble.first->getBufferStart(), Preamble.size());
  memset(const_cast<char*>(PreambleBuffer->getBufferStart()) + Preamble.size(), 
         ' ', PreambleReservedSize - Preamble.size() - 1);
  const_cast<char*>(PreambleBuffer->getBufferEnd())[-1] = '\n';  
  
  // Remap the main source file to the preamble buffer.
  llvm::sys::PathWithStatus MainFilePath(FrontendOpts.Inputs[0].second);
  PreprocessorOpts.addRemappedFile(MainFilePath.str(), PreambleBuffer);
  
  // Tell the compiler invocation to generate a temporary precompiled header.
  FrontendOpts.ProgramAction = frontend::GeneratePCH;
  // FIXME: Set ChainedPCH unconditionally, once it is ready.
  if (::getenv("LIBCLANG_CHAINING"))
    FrontendOpts.ChainedPCH = true;
  // FIXME: Generate the precompiled header into memory?
  FrontendOpts.OutputFile = GetPreamblePCHPath();
  
  // Create the compiler instance to use for building the precompiled preamble.
  CompilerInstance Clang;
  Clang.setInvocation(&PreambleInvocation);
  OriginalSourceFile = Clang.getFrontendOpts().Inputs[0].second;
  
  // Set up diagnostics, capturing all of the diagnostics produced.
  Clang.setDiagnostics(&getDiagnostics());
  CaptureDroppedDiagnostics Capture(CaptureDiagnostics, 
                                    getDiagnostics(),
                                    StoredDiagnostics);
  Clang.setDiagnosticClient(getDiagnostics().getClient());
  
  // Create the target instance.
  Clang.setTarget(TargetInfo::CreateTargetInfo(Clang.getDiagnostics(),
                                               Clang.getTargetOpts()));
  if (!Clang.hasTarget()) {
    Clang.takeDiagnosticClient();
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    if (CreatedPreambleBuffer)
      delete NewPreamble.first;
    if (PreambleTimer)
      PreambleTimer->stopTimer();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    return 0;
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang.getTarget().setForcedLangOptions(Clang.getLangOpts());
  
  assert(Clang.getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_LLVM_IR &&
         "IR inputs not support here!");
  
  // Clear out old caches and data.
  StoredDiagnostics.clear();
  TopLevelDecls.clear();
  TopLevelDeclsInPreamble.clear();
  
  // Create a file manager object to provide access to and cache the filesystem.
  Clang.setFileManager(new FileManager);
  
  // Create the source manager.
  Clang.setSourceManager(new SourceManager(getDiagnostics()));
  
  llvm::OwningPtr<PrecompilePreambleAction> Act;
  Act.reset(new PrecompilePreambleAction(*this));
  if (!Act->BeginSourceFile(Clang, Clang.getFrontendOpts().Inputs[0].second,
                            Clang.getFrontendOpts().Inputs[0].first)) {
    Clang.takeDiagnosticClient();
    Clang.takeInvocation();
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    if (CreatedPreambleBuffer)
      delete NewPreamble.first;
    if (PreambleTimer)
      PreambleTimer->stopTimer();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;

    return 0;
  }
  
  Act->Execute();
  Act->EndSourceFile();
  Clang.takeDiagnosticClient();
  Clang.takeInvocation();
  
  if (Diagnostics->hasErrorOccurred()) {
    // There were errors parsing the preamble, so no precompiled header was
    // generated. Forget that we even tried.
    // FIXME: Should we leave a note for ourselves to try again?
    llvm::sys::Path(FrontendOpts.OutputFile).eraseFromDisk();
    Preamble.clear();
    if (CreatedPreambleBuffer)
      delete NewPreamble.first;
    if (PreambleTimer)
      PreambleTimer->stopTimer();
    TopLevelDeclsInPreamble.clear();
    PreambleRebuildCounter = DefaultPreambleRebuildInterval;
    return 0;
  }
  
  // Keep track of the preamble we precompiled.
  PreambleFile = FrontendOpts.OutputFile;
  NumStoredDiagnosticsInPreamble = StoredDiagnostics.size();
  NumWarningsInPreamble = getDiagnostics().getNumWarnings();
  
  // Keep track of all of the files that the source manager knows about,
  // so we can verify whether they have changed or not.
  FilesInPreamble.clear();
  SourceManager &SourceMgr = Clang.getSourceManager();
  const llvm::MemoryBuffer *MainFileBuffer
    = SourceMgr.getBuffer(SourceMgr.getMainFileID());
  for (SourceManager::fileinfo_iterator F = SourceMgr.fileinfo_begin(),
                                     FEnd = SourceMgr.fileinfo_end();
       F != FEnd;
       ++F) {
    const FileEntry *File = F->second->Entry;
    if (!File || F->second->getRawBuffer() == MainFileBuffer)
      continue;
    
    FilesInPreamble[File->getName()]
      = std::make_pair(F->second->getSize(), File->getModificationTime());
  }
  
  if (PreambleTimer)
    PreambleTimer->stopTimer();
  
  PreambleRebuildCounter = 1;
  return CreatePaddedMainFileBuffer(NewPreamble.first, 
                                    CreatedPreambleBuffer,
                                    PreambleReservedSize,
                                    FrontendOpts.Inputs[0].second);
}

void ASTUnit::RealizeTopLevelDeclsFromPreamble() {
  std::vector<Decl *> Resolved;
  Resolved.reserve(TopLevelDeclsInPreamble.size());
  ExternalASTSource &Source = *getASTContext().getExternalSource();
  for (unsigned I = 0, N = TopLevelDeclsInPreamble.size(); I != N; ++I) {
    // Resolve the declaration ID to an actual declaration, possibly
    // deserializing the declaration in the process.
    Decl *D = Source.GetExternalDecl(TopLevelDeclsInPreamble[I]);
    if (D)
      Resolved.push_back(D);
  }
  TopLevelDeclsInPreamble.clear();
  TopLevelDecls.insert(TopLevelDecls.begin(), Resolved.begin(), Resolved.end());
}

unsigned ASTUnit::getMaxPCHLevel() const {
  if (!getOnlyLocalDecls())
    return Decl::MaxPCHLevel;

  unsigned Result = 0;
  if (isMainFileAST() || SavedMainFileBuffer)
    ++Result;
  return Result;
}

ASTUnit *ASTUnit::LoadFromCompilerInvocation(CompilerInvocation *CI,
                                   llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                             bool OnlyLocalDecls,
                                             bool CaptureDiagnostics,
                                             bool PrecompilePreamble,
                                             bool CompleteTranslationUnit) {
  if (!Diags.getPtr()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    DiagnosticOptions DiagOpts;
    Diags = CompilerInstance::createDiagnostics(DiagOpts, 0, 0);
  }
  
  // Create the AST unit.
  llvm::OwningPtr<ASTUnit> AST;
  AST.reset(new ASTUnit(false));
  AST->Diagnostics = Diags;
  AST->CaptureDiagnostics = CaptureDiagnostics;
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->CompleteTranslationUnit = CompleteTranslationUnit;
  AST->Invocation.reset(CI);
  CI->getPreprocessorOpts().RetainRemappedFileBuffers = true;
  
  if (getenv("LIBCLANG_TIMING"))
    AST->TimerGroup.reset(
                  new llvm::TimerGroup(CI->getFrontendOpts().Inputs[0].second));
  
  
  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  // FIXME: When C++ PCH is ready, allow use of it for a precompiled preamble.
  if (PrecompilePreamble && !CI->getLangOpts().CPlusPlus) {
    AST->PreambleRebuildCounter = 1;
    OverrideMainBuffer = AST->getMainBufferWithPrecompiledPreamble();
  }
  
  llvm::Timer *ParsingTimer = 0;
  if (AST->TimerGroup.get()) {
    ParsingTimer = new llvm::Timer("Initial parse", *AST->TimerGroup);
    ParsingTimer->startTimer();
    AST->Timers.push_back(ParsingTimer);
  }
  
  bool Failed = AST->Parse(OverrideMainBuffer);
  if (ParsingTimer)
    ParsingTimer->stopTimer();
  
  return Failed? 0 : AST.take();
}

ASTUnit *ASTUnit::LoadFromCommandLine(const char **ArgBegin,
                                      const char **ArgEnd,
                                    llvm::IntrusiveRefCntPtr<Diagnostic> Diags,
                                      llvm::StringRef ResourceFilesPath,
                                      bool OnlyLocalDecls,
                                      RemappedFile *RemappedFiles,
                                      unsigned NumRemappedFiles,
                                      bool CaptureDiagnostics,
                                      bool PrecompilePreamble,
                                      bool CompleteTranslationUnit) {
  if (!Diags.getPtr()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    DiagnosticOptions DiagOpts;
    Diags = CompilerInstance::createDiagnostics(DiagOpts, 0, 0);
  }
  
  llvm::SmallVector<const char *, 16> Args;
  Args.push_back("<clang>"); // FIXME: Remove dummy argument.
  Args.insert(Args.end(), ArgBegin, ArgEnd);

  // FIXME: Find a cleaner way to force the driver into restricted modes. We
  // also want to force it to use clang.
  Args.push_back("-fsyntax-only");

  // FIXME: We shouldn't have to pass in the path info.
  driver::Driver TheDriver("clang", llvm::sys::getHostTriple(),
                           "a.out", false, false, *Diags);

  // Don't check that inputs exist, they have been remapped.
  TheDriver.setCheckInputsExist(false);

  llvm::OwningPtr<driver::Compilation> C(
    TheDriver.BuildCompilation(Args.size(), Args.data()));

  // We expect to get back exactly one command job, if we didn't something
  // failed.
  const driver::JobList &Jobs = C->getJobs();
  if (Jobs.size() != 1 || !isa<driver::Command>(Jobs.begin())) {
    llvm::SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    C->PrintJob(OS, C->getJobs(), "; ", true);
    Diags->Report(diag::err_fe_expected_compiler_job) << OS.str();
    return 0;
  }

  const driver::Command *Cmd = cast<driver::Command>(*Jobs.begin());
  if (llvm::StringRef(Cmd->getCreator().getName()) != "clang") {
    Diags->Report(diag::err_fe_expected_clang_command);
    return 0;
  }

  const driver::ArgStringList &CCArgs = Cmd->getArguments();
  llvm::OwningPtr<CompilerInvocation> CI(new CompilerInvocation);
  CompilerInvocation::CreateFromArgs(*CI,
                                     const_cast<const char **>(CCArgs.data()),
                                     const_cast<const char **>(CCArgs.data()) +
                                     CCArgs.size(),
                                     *Diags);

  // Override any files that need remapping
  for (unsigned I = 0; I != NumRemappedFiles; ++I)
    CI->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first,
                                              RemappedFiles[I].second);
  
  // Override the resources path.
  CI->getHeaderSearchOpts().ResourceDir = ResourceFilesPath;

  CI->getFrontendOpts().DisableFree = true;
  return LoadFromCompilerInvocation(CI.take(), Diags, OnlyLocalDecls,
                                    CaptureDiagnostics, PrecompilePreamble,
                                    CompleteTranslationUnit);
}

bool ASTUnit::Reparse(RemappedFile *RemappedFiles, unsigned NumRemappedFiles) {
  if (!Invocation.get())
    return true;
  
  llvm::Timer *ReparsingTimer = 0;
  if (TimerGroup.get()) {
    ReparsingTimer = new llvm::Timer("Reparse", *TimerGroup);
    ReparsingTimer->startTimer();
    Timers.push_back(ReparsingTimer);
  }
  
  // Remap files.
  Invocation->getPreprocessorOpts().clearRemappedFiles();
  for (unsigned I = 0; I != NumRemappedFiles; ++I)
    Invocation->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first,
                                                      RemappedFiles[I].second);
  
  // If we have a preamble file lying around, or if we might try to
  // build a precompiled preamble, do so now.
  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  if (!PreambleFile.empty() || PreambleRebuildCounter > 0)
    OverrideMainBuffer = getMainBufferWithPrecompiledPreamble();
    
  // Clear out the diagnostics state.
  if (!OverrideMainBuffer)
    getDiagnostics().Reset();
  
  // Parse the sources
  bool Result = Parse(OverrideMainBuffer);  
  if (ReparsingTimer)
    ReparsingTimer->stopTimer();
  return Result;
}

void ASTUnit::CodeComplete(llvm::StringRef File, unsigned Line, unsigned Column,
                           RemappedFile *RemappedFiles, 
                           unsigned NumRemappedFiles,
                           bool IncludeMacros, 
                           bool IncludeCodePatterns,
                           CodeCompleteConsumer &Consumer,
                           Diagnostic &Diag, LangOptions &LangOpts,
                           SourceManager &SourceMgr, FileManager &FileMgr,
                   llvm::SmallVectorImpl<StoredDiagnostic> &StoredDiagnostics) {
  if (!Invocation.get())
    return;

  llvm::Timer *CompletionTimer = 0;
  if (TimerGroup.get()) {
    llvm::SmallString<128> TimerName;
    llvm::raw_svector_ostream TimerNameOut(TimerName);
    TimerNameOut << "Code completion @ " << File << ":" << Line << ":" 
                 << Column;
    CompletionTimer = new llvm::Timer(TimerNameOut.str(), *TimerGroup);
    CompletionTimer->startTimer();
    Timers.push_back(CompletionTimer);
  }

  CompilerInvocation CCInvocation(*Invocation);
  FrontendOptions &FrontendOpts = CCInvocation.getFrontendOpts();
  PreprocessorOptions &PreprocessorOpts = CCInvocation.getPreprocessorOpts();

  FrontendOpts.ShowMacrosInCodeCompletion = IncludeMacros;
  FrontendOpts.ShowCodePatternsInCodeCompletion = IncludeCodePatterns;
  FrontendOpts.CodeCompletionAt.FileName = File;
  FrontendOpts.CodeCompletionAt.Line = Line;
  FrontendOpts.CodeCompletionAt.Column = Column;

  // Turn on spell-checking when performing code completion. It leads
  // to better results.
  unsigned SpellChecking = CCInvocation.getLangOpts().SpellChecking;
  CCInvocation.getLangOpts().SpellChecking = 1;

  // Set the language options appropriately.
  LangOpts = CCInvocation.getLangOpts();

  CompilerInstance Clang;
  Clang.setInvocation(&CCInvocation);
  OriginalSourceFile = Clang.getFrontendOpts().Inputs[0].second;
    
  // Set up diagnostics, capturing any diagnostics produced.
  Clang.setDiagnostics(&Diag);
  CaptureDroppedDiagnostics Capture(true, 
                                    Clang.getDiagnostics(),
                                    StoredDiagnostics);
  Clang.setDiagnosticClient(Diag.getClient());
  
  // Create the target instance.
  Clang.setTarget(TargetInfo::CreateTargetInfo(Clang.getDiagnostics(),
                                               Clang.getTargetOpts()));
  if (!Clang.hasTarget()) {
    Clang.takeDiagnosticClient();
    Clang.takeInvocation();
  }
  
  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang.getTarget().setForcedLangOptions(Clang.getLangOpts());
  
  assert(Clang.getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_AST &&
         "FIXME: AST inputs not yet supported here!");
  assert(Clang.getFrontendOpts().Inputs[0].first != IK_LLVM_IR &&
         "IR inputs not support here!");

  
  // Use the source and file managers that we were given.
  Clang.setFileManager(&FileMgr);
  Clang.setSourceManager(&SourceMgr);

  // Remap files.
  PreprocessorOpts.clearRemappedFiles();
  PreprocessorOpts.RetainRemappedFileBuffers = true;
  for (unsigned I = 0; I != NumRemappedFiles; ++I)
    PreprocessorOpts.addRemappedFile(RemappedFiles[I].first,
                                     RemappedFiles[I].second);
  
  // Use the code completion consumer we were given.
  Clang.setCodeCompletionConsumer(&Consumer);

  // If we have a precompiled preamble, try to use it. We only allow
  // the use of the precompiled preamble if we're if the completion
  // point is within the main file, after the end of the precompiled
  // preamble.
  llvm::MemoryBuffer *OverrideMainBuffer = 0;
  if (!PreambleFile.empty()) {
    using llvm::sys::FileStatus;
    llvm::sys::PathWithStatus CompleteFilePath(File);
    llvm::sys::PathWithStatus MainPath(OriginalSourceFile);
    if (const FileStatus *CompleteFileStatus = CompleteFilePath.getFileStatus())
      if (const FileStatus *MainStatus = MainPath.getFileStatus())
        if (CompleteFileStatus->getUniqueID() == MainStatus->getUniqueID())
          OverrideMainBuffer = getMainBufferWithPrecompiledPreamble(false, 
                                                                    Line);
  }

  // If the main file has been overridden due to the use of a preamble,
  // make that override happen and introduce the preamble.
  if (OverrideMainBuffer) {
    PreprocessorOpts.addRemappedFile(OriginalSourceFile, OverrideMainBuffer);
    PreprocessorOpts.PrecompiledPreambleBytes.first = Preamble.size();
    PreprocessorOpts.PrecompiledPreambleBytes.second
                                                    = PreambleEndsAtStartOfLine;
    PreprocessorOpts.ImplicitPCHInclude = PreambleFile;
    PreprocessorOpts.DisablePCHValidation = true;
    
    // The stored diagnostics have the old source manager. Copy them
    // to our output set of stored diagnostics, updating the source
    // manager to the one we were given.
    for (unsigned I = 0, N = this->StoredDiagnostics.size(); I != N; ++I) {
      StoredDiagnostics.push_back(this->StoredDiagnostics[I]);
      FullSourceLoc Loc(StoredDiagnostics[I].getLocation(), SourceMgr);
      StoredDiagnostics[I].setLocation(Loc);
    }
  }

  llvm::OwningPtr<SyntaxOnlyAction> Act;
  Act.reset(new SyntaxOnlyAction);
  if (Act->BeginSourceFile(Clang, Clang.getFrontendOpts().Inputs[0].second,
                           Clang.getFrontendOpts().Inputs[0].first)) {
    Act->Execute();
    Act->EndSourceFile();
  }

  if (CompletionTimer)
    CompletionTimer->stopTimer();
  
  // Steal back our resources. 
  delete OverrideMainBuffer;
  Clang.takeFileManager();
  Clang.takeSourceManager();
  Clang.takeInvocation();
  Clang.takeDiagnosticClient();
  Clang.takeCodeCompletionConsumer();
  CCInvocation.getLangOpts().SpellChecking = SpellChecking;
}
