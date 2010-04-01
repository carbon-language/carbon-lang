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
#include "clang/Frontend/PCHReader.h"
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
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
using namespace clang;

ASTUnit::ASTUnit(Diagnostic &Diag, bool _MainFileIsAST)
  : SourceMgr(Diag), MainFileIsAST(_MainFileIsAST), 
    ConcurrencyCheckValue(CheckUnlocked) {
}
ASTUnit::~ASTUnit() {
  ConcurrencyCheckValue = CheckLocked;
  for (unsigned I = 0, N = TemporaryFiles.size(); I != N; ++I)
    TemporaryFiles[I].eraseFromDisk();
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

  virtual bool ReadPredefinesBuffer(llvm::StringRef PCHPredef,
                                    FileID PCHBufferID,
                                    llvm::StringRef OriginalFileName,
                                    std::string &SuggestedPredefines) {
    Predefines = PCHPredef;
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
                                  Diagnostic &Diags,
                                  bool OnlyLocalDecls,
                                  RemappedFile *RemappedFiles,
                                  unsigned NumRemappedFiles,
                                  bool CaptureDiagnostics) {
  llvm::OwningPtr<ASTUnit> AST(new ASTUnit(Diags, true));
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->HeaderInfo.reset(new HeaderSearch(AST->getFileManager()));

  // If requested, capture diagnostics in the ASTUnit.
  CaptureDroppedDiagnostics Capture(CaptureDiagnostics, Diags, 
                                    AST->Diagnostics);

  for (unsigned I = 0; I != NumRemappedFiles; ++I) {
    // Create the file entry for the file that we're mapping from.
    const FileEntry *FromFile
      = AST->getFileManager().getVirtualFile(RemappedFiles[I].first,
                                    RemappedFiles[I].second->getBufferSize(),
                                             0);
    if (!FromFile) {
      Diags.Report(diag::err_fe_remap_missing_from_file)
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
                             Diags));
  Reader->setListener(new PCHInfoCollector(LangInfo, HeaderInfo, TargetTriple,
                                           Predefines, Counter));

  switch (Reader->ReadPCH(Filename)) {
  case PCHReader::Success:
    break;

  case PCHReader::Failure:
  case PCHReader::IgnorePCH:
    Diags.Report(diag::err_fe_unable_to_load_pch);
    return NULL;
  }

  AST->OriginalSourceFile = Reader->getOriginalSourceFile();

  // PCH loaded successfully. Now create the preprocessor.

  // Get information about the target being compiled for.
  //
  // FIXME: This is broken, we should store the TargetOptions in the PCH.
  TargetOptions TargetOpts;
  TargetOpts.ABI = "";
  TargetOpts.CPU = "";
  TargetOpts.Features.clear();
  TargetOpts.Triple = TargetTriple;
  AST->Target.reset(TargetInfo::CreateTargetInfo(Diags, TargetOpts));
  AST->PP.reset(new Preprocessor(Diags, LangInfo, *AST->Target.get(),
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
                                /* FreeMemory = */ false,
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
    for (DeclGroupRef::iterator it = D.begin(), ie = D.end(); it != ie; ++it)
      Unit.getTopLevelDecls().push_back(*it);
  }
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
};

}

ASTUnit *ASTUnit::LoadFromCompilerInvocation(CompilerInvocation *CI,
                                             Diagnostic &Diags,
                                             bool OnlyLocalDecls,
                                             bool CaptureDiagnostics) {
  // Create the compiler instance to use for building the AST.
  CompilerInstance Clang;
  llvm::OwningPtr<ASTUnit> AST;
  llvm::OwningPtr<TopLevelDeclTrackerAction> Act;

  Clang.setInvocation(CI);

  Clang.setDiagnostics(&Diags);
  Clang.setDiagnosticClient(Diags.getClient());

  // Create the target instance.
  Clang.setTarget(TargetInfo::CreateTargetInfo(Clang.getDiagnostics(),
                                               Clang.getTargetOpts()));
  if (!Clang.hasTarget()) {
    Clang.takeDiagnosticClient();
    Clang.takeDiagnostics();
    return 0;
  }

  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang.getTarget().setForcedLangOptions(Clang.getLangOpts());

  assert(Clang.getFrontendOpts().Inputs.size() == 1 &&
         "Invocation must have exactly one source file!");
  assert(Clang.getFrontendOpts().Inputs[0].first != FrontendOptions::IK_AST &&
         "FIXME: AST inputs not yet supported here!");

  // Create the AST unit.
  AST.reset(new ASTUnit(Diags, false));
  AST->OnlyLocalDecls = OnlyLocalDecls;
  AST->OriginalSourceFile = Clang.getFrontendOpts().Inputs[0].second;

  // Capture any diagnostics that would otherwise be dropped.
  CaptureDroppedDiagnostics Capture(CaptureDiagnostics, 
                                    Clang.getDiagnostics(),
                                    AST->Diagnostics);

  // Create a file manager object to provide access to and cache the filesystem.
  Clang.setFileManager(&AST->getFileManager());

  // Create the source manager.
  Clang.setSourceManager(&AST->getSourceManager());

  // Create the preprocessor.
  Clang.createPreprocessor();

  Act.reset(new TopLevelDeclTrackerAction(*AST));
  if (!Act->BeginSourceFile(Clang, Clang.getFrontendOpts().Inputs[0].second,
                           /*IsAST=*/false))
    goto error;

  Act->Execute();

  // Steal the created target, context, and preprocessor, and take back the
  // source and file managers.
  AST->Ctx.reset(Clang.takeASTContext());
  AST->PP.reset(Clang.takePreprocessor());
  Clang.takeSourceManager();
  Clang.takeFileManager();
  AST->Target.reset(Clang.takeTarget());

  Act->EndSourceFile();

  Clang.takeDiagnosticClient();
  Clang.takeDiagnostics();
  Clang.takeInvocation();

  AST->Invocation.reset(Clang.takeInvocation());
  return AST.take();

error:
  Clang.takeSourceManager();
  Clang.takeFileManager();
  Clang.takeDiagnosticClient();
  Clang.takeDiagnostics();
  return 0;
}

ASTUnit *ASTUnit::LoadFromCommandLine(const char **ArgBegin,
                                      const char **ArgEnd,
                                      Diagnostic &Diags,
                                      llvm::StringRef ResourceFilesPath,
                                      bool OnlyLocalDecls,
                                      RemappedFile *RemappedFiles,
                                      unsigned NumRemappedFiles,
                                      bool CaptureDiagnostics) {
  llvm::SmallVector<const char *, 16> Args;
  Args.push_back("<clang>"); // FIXME: Remove dummy argument.
  Args.insert(Args.end(), ArgBegin, ArgEnd);

  // FIXME: Find a cleaner way to force the driver into restricted modes. We
  // also want to force it to use clang.
  Args.push_back("-fsyntax-only");

  // FIXME: We shouldn't have to pass in the path info.
  driver::Driver TheDriver("clang", "/", llvm::sys::getHostTriple(),
                           "a.out", false, false, Diags);

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
    Diags.Report(diag::err_fe_expected_compiler_job) << OS.str();
    return 0;
  }

  const driver::Command *Cmd = cast<driver::Command>(*Jobs.begin());
  if (llvm::StringRef(Cmd->getCreator().getName()) != "clang") {
    Diags.Report(diag::err_fe_expected_clang_command);
    return 0;
  }

  const driver::ArgStringList &CCArgs = Cmd->getArguments();
  llvm::OwningPtr<CompilerInvocation> CI(new CompilerInvocation);
  CompilerInvocation::CreateFromArgs(*CI, (const char**) CCArgs.data(),
                                     (const char**) CCArgs.data()+CCArgs.size(),
                                     Diags);

  // Override any files that need remapping
  for (unsigned I = 0; I != NumRemappedFiles; ++I)
    CI->getPreprocessorOpts().addRemappedFile(RemappedFiles[I].first,
                                              RemappedFiles[I].second);
  
  // Override the resources path.
  CI->getHeaderSearchOpts().ResourceDir = ResourceFilesPath;

  CI->getFrontendOpts().DisableFree = true;
  return LoadFromCompilerInvocation(CI.take(), Diags, OnlyLocalDecls,
                                    CaptureDiagnostics);
}
