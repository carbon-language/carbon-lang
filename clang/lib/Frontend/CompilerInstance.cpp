//===--- CompilerInstance.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PTHManager.h"
#include "clang/Frontend/ChainedDiagnosticClient.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/LogDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "clang/Frontend/Utils.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/system_error.h"
#include "llvm/Config/config.h"
using namespace clang;

CompilerInstance::CompilerInstance()
  : Invocation(new CompilerInvocation()), ModuleManager(0) {
}

CompilerInstance::~CompilerInstance() {
}

void CompilerInstance::setInvocation(CompilerInvocation *Value) {
  Invocation = Value;
}

void CompilerInstance::setDiagnostics(Diagnostic *Value) {
  Diagnostics = Value;
}

void CompilerInstance::setTarget(TargetInfo *Value) {
  Target = Value;
}

void CompilerInstance::setFileManager(FileManager *Value) {
  FileMgr = Value;
}

void CompilerInstance::setSourceManager(SourceManager *Value) { 
  SourceMgr = Value;
}

void CompilerInstance::setPreprocessor(Preprocessor *Value) { PP = Value; }

void CompilerInstance::setASTContext(ASTContext *Value) { Context = Value; }

void CompilerInstance::setSema(Sema *S) {
  TheSema.reset(S);
}

void CompilerInstance::setASTConsumer(ASTConsumer *Value) {
  Consumer.reset(Value);
}

void CompilerInstance::setCodeCompletionConsumer(CodeCompleteConsumer *Value) {
  CompletionConsumer.reset(Value);
}

// Diagnostics
static void SetUpBuildDumpLog(const DiagnosticOptions &DiagOpts,
                              unsigned argc, const char* const *argv,
                              Diagnostic &Diags) {
  std::string ErrorInfo;
  llvm::OwningPtr<raw_ostream> OS(
    new llvm::raw_fd_ostream(DiagOpts.DumpBuildInformation.c_str(), ErrorInfo));
  if (!ErrorInfo.empty()) {
    Diags.Report(diag::err_fe_unable_to_open_logfile)
                 << DiagOpts.DumpBuildInformation << ErrorInfo;
    return;
  }

  (*OS) << "clang -cc1 command line arguments: ";
  for (unsigned i = 0; i != argc; ++i)
    (*OS) << argv[i] << ' ';
  (*OS) << '\n';

  // Chain in a diagnostic client which will log the diagnostics.
  DiagnosticClient *Logger =
    new TextDiagnosticPrinter(*OS.take(), DiagOpts, /*OwnsOutputStream=*/true);
  Diags.setClient(new ChainedDiagnosticClient(Diags.takeClient(), Logger));
}

static void SetUpDiagnosticLog(const DiagnosticOptions &DiagOpts,
                               const CodeGenOptions *CodeGenOpts,
                               Diagnostic &Diags) {
  std::string ErrorInfo;
  bool OwnsStream = false;
  raw_ostream *OS = &llvm::errs();
  if (DiagOpts.DiagnosticLogFile != "-") {
    // Create the output stream.
    llvm::raw_fd_ostream *FileOS(
      new llvm::raw_fd_ostream(DiagOpts.DiagnosticLogFile.c_str(),
                               ErrorInfo, llvm::raw_fd_ostream::F_Append));
    if (!ErrorInfo.empty()) {
      Diags.Report(diag::warn_fe_cc_log_diagnostics_failure)
        << DiagOpts.DumpBuildInformation << ErrorInfo;
    } else {
      FileOS->SetUnbuffered();
      FileOS->SetUseAtomicWrites(true);
      OS = FileOS;
      OwnsStream = true;
    }
  }

  // Chain in the diagnostic client which will log the diagnostics.
  LogDiagnosticPrinter *Logger = new LogDiagnosticPrinter(*OS, DiagOpts,
                                                          OwnsStream);
  if (CodeGenOpts)
    Logger->setDwarfDebugFlags(CodeGenOpts->DwarfDebugFlags);
  Diags.setClient(new ChainedDiagnosticClient(Diags.takeClient(), Logger));
}

void CompilerInstance::createDiagnostics(int Argc, const char* const *Argv,
                                         DiagnosticClient *Client,
                                         bool ShouldOwnClient) {
  Diagnostics = createDiagnostics(getDiagnosticOpts(), Argc, Argv, Client,
                                  ShouldOwnClient, &getCodeGenOpts());
}

llvm::IntrusiveRefCntPtr<Diagnostic> 
CompilerInstance::createDiagnostics(const DiagnosticOptions &Opts,
                                    int Argc, const char* const *Argv,
                                    DiagnosticClient *Client,
                                    bool ShouldOwnClient,
                                    const CodeGenOptions *CodeGenOpts) {
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(new Diagnostic(DiagID));

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  if (Client)
    Diags->setClient(Client, ShouldOwnClient);
  else
    Diags->setClient(new TextDiagnosticPrinter(llvm::errs(), Opts));

  // Chain in -verify checker, if requested.
  if (Opts.VerifyDiagnostics) 
    Diags->setClient(new VerifyDiagnosticsClient(*Diags));

  // Chain in -diagnostic-log-file dumper, if requested.
  if (!Opts.DiagnosticLogFile.empty())
    SetUpDiagnosticLog(Opts, CodeGenOpts, *Diags);
  
  if (!Opts.DumpBuildInformation.empty())
    SetUpBuildDumpLog(Opts, Argc, Argv, *Diags);

  // Configure our handling of diagnostics.
  ProcessWarningOptions(*Diags, Opts);

  return Diags;
}

// File Manager

void CompilerInstance::createFileManager() {
  FileMgr = new FileManager(getFileSystemOpts());
}

// Source Manager

void CompilerInstance::createSourceManager(FileManager &FileMgr) {
  SourceMgr = new SourceManager(getDiagnostics(), FileMgr);
}

// Preprocessor

void CompilerInstance::createPreprocessor() {
  const PreprocessorOptions &PPOpts = getPreprocessorOpts();
  
  // Create a PTH manager if we are using some form of a token cache.
  PTHManager *PTHMgr = 0;
  if (!PPOpts.TokenCache.empty())
    PTHMgr = PTHManager::Create(PPOpts.TokenCache, getDiagnostics());
  
  // Create the Preprocessor.
  HeaderSearch *HeaderInfo = new HeaderSearch(getFileManager());
  PP = new Preprocessor(getDiagnostics(), getLangOpts(), &getTarget(),
                        getSourceManager(), *HeaderInfo, *this, PTHMgr,
                        /*OwnsHeaderSearch=*/true);
  
  // Note that this is different then passing PTHMgr to Preprocessor's ctor.
  // That argument is used as the IdentifierInfoLookup argument to
  // IdentifierTable's ctor.
  if (PTHMgr) {
    PTHMgr->setPreprocessor(&*PP);
    PP->setPTHManager(PTHMgr);
  }
  
  if (PPOpts.DetailedRecord)
    PP->createPreprocessingRecord(
                                  PPOpts.DetailedRecordIncludesNestedMacroExpansions);
  
  InitializePreprocessor(*PP, PPOpts, getHeaderSearchOpts(), getFrontendOpts());
  
  // Set up the module path, including the hash for the
  // module-creation options.
  llvm::SmallString<256> SpecificModuleCache(
                           getHeaderSearchOpts().ModuleCachePath);
  if (!getHeaderSearchOpts().DisableModuleHash)
    llvm::sys::path::append(SpecificModuleCache, 
                            getInvocation().getModuleHash());
  PP->getHeaderSearchInfo().setModuleCachePath(SpecificModuleCache);
  
  // Handle generating dependencies, if requested.
  const DependencyOutputOptions &DepOpts = getDependencyOutputOpts();
  if (!DepOpts.OutputFile.empty())
    AttachDependencyFileGen(*PP, DepOpts);
  
  // Handle generating header include information, if requested.
  if (DepOpts.ShowHeaderIncludes)
    AttachHeaderIncludeGen(*PP);
  if (!DepOpts.HeaderIncludeOutputFile.empty()) {
    StringRef OutputPath = DepOpts.HeaderIncludeOutputFile;
    if (OutputPath == "-")
      OutputPath = "";
    AttachHeaderIncludeGen(*PP, /*ShowAllHeaders=*/true, OutputPath,
                           /*ShowDepth=*/false);
  }
}

// ASTContext

void CompilerInstance::createASTContext() {
  Preprocessor &PP = getPreprocessor();
  Context = new ASTContext(getLangOpts(), PP.getSourceManager(),
                           &getTarget(), PP.getIdentifierTable(),
                           PP.getSelectorTable(), PP.getBuiltinInfo(),
                           /*size_reserve=*/ 0);
}

// ExternalASTSource

void CompilerInstance::createPCHExternalASTSource(StringRef Path,
                                                  bool DisablePCHValidation,
                                                  bool DisableStatCache,
                                                 void *DeserializationListener){
  llvm::OwningPtr<ExternalASTSource> Source;
  bool Preamble = getPreprocessorOpts().PrecompiledPreambleBytes.first != 0;
  Source.reset(createPCHExternalASTSource(Path, getHeaderSearchOpts().Sysroot,
                                          DisablePCHValidation, 
                                          DisableStatCache,
                                          getPreprocessor(), getASTContext(),
                                          DeserializationListener,
                                          Preamble));
  ModuleManager = static_cast<ASTReader*>(Source.get());
  getASTContext().setExternalSource(Source);
}

ExternalASTSource *
CompilerInstance::createPCHExternalASTSource(StringRef Path,
                                             const std::string &Sysroot,
                                             bool DisablePCHValidation,
                                             bool DisableStatCache,
                                             Preprocessor &PP,
                                             ASTContext &Context,
                                             void *DeserializationListener,
                                             bool Preamble) {
  llvm::OwningPtr<ASTReader> Reader;
  Reader.reset(new ASTReader(PP, Context,
                             Sysroot.empty() ? "" : Sysroot.c_str(),
                             DisablePCHValidation, DisableStatCache));

  Reader->setDeserializationListener(
            static_cast<ASTDeserializationListener *>(DeserializationListener));
  switch (Reader->ReadAST(Path, 
                          Preamble ? serialization::MK_Preamble 
                                   : serialization::MK_PCH)) {
  case ASTReader::Success:
    // Set the predefines buffer as suggested by the PCH reader. Typically, the
    // predefines buffer will be empty.
    PP.setPredefines(Reader->getSuggestedPredefines());
    return Reader.take();

  case ASTReader::Failure:
    // Unrecoverable failure: don't even try to process the input file.
    break;

  case ASTReader::IgnorePCH:
    // No suitable PCH file could be found. Return an error.
    break;
  }

  return 0;
}

// Code Completion

static bool EnableCodeCompletion(Preprocessor &PP, 
                                 const std::string &Filename,
                                 unsigned Line,
                                 unsigned Column) {
  // Tell the source manager to chop off the given file at a specific
  // line and column.
  const FileEntry *Entry = PP.getFileManager().getFile(Filename);
  if (!Entry) {
    PP.getDiagnostics().Report(diag::err_fe_invalid_code_complete_file)
      << Filename;
    return true;
  }

  // Truncate the named file at the given line/column.
  PP.SetCodeCompletionPoint(Entry, Line, Column);
  return false;
}

void CompilerInstance::createCodeCompletionConsumer() {
  const ParsedSourceLocation &Loc = getFrontendOpts().CodeCompletionAt;
  if (!CompletionConsumer) {
    CompletionConsumer.reset(
      createCodeCompletionConsumer(getPreprocessor(),
                                   Loc.FileName, Loc.Line, Loc.Column,
                                   getFrontendOpts().ShowMacrosInCodeCompletion,
                             getFrontendOpts().ShowCodePatternsInCodeCompletion,
                           getFrontendOpts().ShowGlobalSymbolsInCodeCompletion,
                                   llvm::outs()));
    if (!CompletionConsumer)
      return;
  } else if (EnableCodeCompletion(getPreprocessor(), Loc.FileName,
                                  Loc.Line, Loc.Column)) {
    CompletionConsumer.reset();
    return;
  }

  if (CompletionConsumer->isOutputBinary() &&
      llvm::sys::Program::ChangeStdoutToBinary()) {
    getPreprocessor().getDiagnostics().Report(diag::err_fe_stdout_binary);
    CompletionConsumer.reset();
  }
}

void CompilerInstance::createFrontendTimer() {
  FrontendTimer.reset(new llvm::Timer("Clang front-end timer"));
}

CodeCompleteConsumer *
CompilerInstance::createCodeCompletionConsumer(Preprocessor &PP,
                                               const std::string &Filename,
                                               unsigned Line,
                                               unsigned Column,
                                               bool ShowMacros,
                                               bool ShowCodePatterns,
                                               bool ShowGlobals,
                                               raw_ostream &OS) {
  if (EnableCodeCompletion(PP, Filename, Line, Column))
    return 0;

  // Set up the creation routine for code-completion.
  return new PrintingCodeCompleteConsumer(ShowMacros, ShowCodePatterns, 
                                          ShowGlobals, OS);
}

void CompilerInstance::createSema(TranslationUnitKind TUKind,
                                  CodeCompleteConsumer *CompletionConsumer) {
  TheSema.reset(new Sema(getPreprocessor(), getASTContext(), getASTConsumer(),
                         TUKind, CompletionConsumer));
}

// Output Files

void CompilerInstance::addOutputFile(const OutputFile &OutFile) {
  assert(OutFile.OS && "Attempt to add empty stream to output list!");
  OutputFiles.push_back(OutFile);
}

void CompilerInstance::clearOutputFiles(bool EraseFiles) {
  for (std::list<OutputFile>::iterator
         it = OutputFiles.begin(), ie = OutputFiles.end(); it != ie; ++it) {
    delete it->OS;
    if (!it->TempFilename.empty()) {
      if (EraseFiles) {
        bool existed;
        llvm::sys::fs::remove(it->TempFilename, existed);
      } else {
        llvm::SmallString<128> NewOutFile(it->Filename);

        // If '-working-directory' was passed, the output filename should be
        // relative to that.
        FileMgr->FixupRelativePath(NewOutFile);
        if (llvm::error_code ec = llvm::sys::fs::rename(it->TempFilename,
                                                        NewOutFile.str())) {
          getDiagnostics().Report(diag::err_fe_unable_to_rename_temp)
            << it->TempFilename << it->Filename << ec.message();

          bool existed;
          llvm::sys::fs::remove(it->TempFilename, existed);
        }
      }
    } else if (!it->Filename.empty() && EraseFiles)
      llvm::sys::Path(it->Filename).eraseFromDisk();
      
  }
  OutputFiles.clear();
}

llvm::raw_fd_ostream *
CompilerInstance::createDefaultOutputFile(bool Binary,
                                          StringRef InFile,
                                          StringRef Extension) {
  return createOutputFile(getFrontendOpts().OutputFile, Binary,
                          /*RemoveFileOnSignal=*/true, InFile, Extension);
}

llvm::raw_fd_ostream *
CompilerInstance::createOutputFile(StringRef OutputPath,
                                   bool Binary, bool RemoveFileOnSignal,
                                   StringRef InFile,
                                   StringRef Extension,
                                   bool UseTemporary) {
  std::string Error, OutputPathName, TempPathName;
  llvm::raw_fd_ostream *OS = createOutputFile(OutputPath, Error, Binary,
                                              RemoveFileOnSignal,
                                              InFile, Extension,
                                              UseTemporary,
                                              &OutputPathName,
                                              &TempPathName);
  if (!OS) {
    getDiagnostics().Report(diag::err_fe_unable_to_open_output)
      << OutputPath << Error;
    return 0;
  }

  // Add the output file -- but don't try to remove "-", since this means we are
  // using stdin.
  addOutputFile(OutputFile((OutputPathName != "-") ? OutputPathName : "",
                TempPathName, OS));

  return OS;
}

llvm::raw_fd_ostream *
CompilerInstance::createOutputFile(StringRef OutputPath,
                                   std::string &Error,
                                   bool Binary,
                                   bool RemoveFileOnSignal,
                                   StringRef InFile,
                                   StringRef Extension,
                                   bool UseTemporary,
                                   std::string *ResultPathName,
                                   std::string *TempPathName) {
  std::string OutFile, TempFile;
  if (!OutputPath.empty()) {
    OutFile = OutputPath;
  } else if (InFile == "-") {
    OutFile = "-";
  } else if (!Extension.empty()) {
    llvm::sys::Path Path(InFile);
    Path.eraseSuffix();
    Path.appendSuffix(Extension);
    OutFile = Path.str();
  } else {
    OutFile = "-";
  }

  llvm::OwningPtr<llvm::raw_fd_ostream> OS;
  std::string OSFile;

  if (UseTemporary && OutFile != "-") {
    llvm::sys::Path OutPath(OutFile);
    // Only create the temporary if we can actually write to OutPath, otherwise
    // we want to fail early.
    bool Exists;
    if ((llvm::sys::fs::exists(OutPath.str(), Exists) || !Exists) ||
        (OutPath.isRegularFile() && OutPath.canWrite())) {
      // Create a temporary file.
      llvm::SmallString<128> TempPath;
      TempPath = OutFile;
      TempPath += "-%%%%%%%%";
      int fd;
      if (llvm::sys::fs::unique_file(TempPath.str(), fd, TempPath,
                               /*makeAbsolute=*/false) == llvm::errc::success) {
        OS.reset(new llvm::raw_fd_ostream(fd, /*shouldClose=*/true));
        OSFile = TempFile = TempPath.str();
      }
    }
  }

  if (!OS) {
    OSFile = OutFile;
    OS.reset(
      new llvm::raw_fd_ostream(OSFile.c_str(), Error,
                               (Binary ? llvm::raw_fd_ostream::F_Binary : 0)));
    if (!Error.empty())
      return 0;
  }

  // Make sure the out stream file gets removed if we crash.
  if (RemoveFileOnSignal)
    llvm::sys::RemoveFileOnSignal(llvm::sys::Path(OSFile));

  if (ResultPathName)
    *ResultPathName = OutFile;
  if (TempPathName)
    *TempPathName = TempFile;

  return OS.take();
}

// Initialization Utilities

bool CompilerInstance::InitializeSourceManager(StringRef InputFile) {
  return InitializeSourceManager(InputFile, getDiagnostics(), getFileManager(),
                                 getSourceManager(), getFrontendOpts());
}

bool CompilerInstance::InitializeSourceManager(StringRef InputFile,
                                               Diagnostic &Diags,
                                               FileManager &FileMgr,
                                               SourceManager &SourceMgr,
                                               const FrontendOptions &Opts) {
  // Figure out where to get and map in the main file, unless it's already
  // been created (e.g., by a precompiled preamble).
  if (!SourceMgr.getMainFileID().isInvalid()) {
    // Do nothing: the main file has already been set.
  } else if (InputFile != "-") {
    const FileEntry *File = FileMgr.getFile(InputFile);
    if (!File) {
      Diags.Report(diag::err_fe_error_reading) << InputFile;
      return false;
    }
    SourceMgr.createMainFileID(File);
  } else {
    llvm::OwningPtr<llvm::MemoryBuffer> SB;
    if (llvm::MemoryBuffer::getSTDIN(SB)) {
      // FIXME: Give ec.message() in this diag.
      Diags.Report(diag::err_fe_error_reading_stdin);
      return false;
    }
    const FileEntry *File = FileMgr.getVirtualFile(SB->getBufferIdentifier(),
                                                   SB->getBufferSize(), 0);
    SourceMgr.createMainFileID(File);
    SourceMgr.overrideFileContents(File, SB.take());
  }

  assert(!SourceMgr.getMainFileID().isInvalid() &&
         "Couldn't establish MainFileID!");
  return true;
}

// High-Level Operations

bool CompilerInstance::ExecuteAction(FrontendAction &Act) {
  assert(hasDiagnostics() && "Diagnostics engine is not initialized!");
  assert(!getFrontendOpts().ShowHelp && "Client must handle '-help'!");
  assert(!getFrontendOpts().ShowVersion && "Client must handle '-version'!");

  // FIXME: Take this as an argument, once all the APIs we used have moved to
  // taking it as an input instead of hard-coding llvm::errs.
  raw_ostream &OS = llvm::errs();

  // Create the target instance.
  setTarget(TargetInfo::CreateTargetInfo(getDiagnostics(), getTargetOpts()));
  if (!hasTarget())
    return false;

  // Inform the target of the language options.
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  getTarget().setForcedLangOptions(getLangOpts());

  // Validate/process some options.
  if (getHeaderSearchOpts().Verbose)
    OS << "clang -cc1 version " CLANG_VERSION_STRING
       << " based upon " << PACKAGE_STRING
       << " hosted on " << llvm::sys::getHostTriple() << "\n";

  if (getFrontendOpts().ShowTimers)
    createFrontendTimer();

  if (getFrontendOpts().ShowStats)
    llvm::EnableStatistics();
    
  for (unsigned i = 0, e = getFrontendOpts().Inputs.size(); i != e; ++i) {
    const std::string &InFile = getFrontendOpts().Inputs[i].second;

    // Reset the ID tables if we are reusing the SourceManager.
    if (hasSourceManager())
      getSourceManager().clearIDTables();

    if (Act.BeginSourceFile(*this, InFile, getFrontendOpts().Inputs[i].first)) {
      Act.Execute();
      Act.EndSourceFile();
    }
  }

  if (getDiagnosticOpts().ShowCarets) {
    // We can have multiple diagnostics sharing one diagnostic client.
    // Get the total number of warnings/errors from the client.
    unsigned NumWarnings = getDiagnostics().getClient()->getNumWarnings();
    unsigned NumErrors = getDiagnostics().getClient()->getNumErrors();
    
    if (NumWarnings)
      OS << NumWarnings << " warning" << (NumWarnings == 1 ? "" : "s");
    if (NumWarnings && NumErrors)
      OS << " and ";
    if (NumErrors)
      OS << NumErrors << " error" << (NumErrors == 1 ? "" : "s");
    if (NumWarnings || NumErrors)
      OS << " generated.\n";
  }

  if (getFrontendOpts().ShowStats && hasFileManager()) {
    getFileManager().PrintStats();
    OS << "\n";
  }

  return !getDiagnostics().getClient()->getNumErrors();
}

/// \brief Determine the appropriate source input kind based on language
/// options.
static InputKind getSourceInputKindFromOptions(const LangOptions &LangOpts) {
  if (LangOpts.OpenCL)
    return IK_OpenCL;
  if (LangOpts.CUDA)
    return IK_CUDA;
  if (LangOpts.ObjC1)
    return LangOpts.CPlusPlus? IK_ObjCXX : IK_ObjC;
  return LangOpts.CPlusPlus? IK_CXX : IK_C;
}

/// \brief Compile a module file for the given module name with the given
/// umbrella header, using the options provided by the importing compiler
/// instance.
static void compileModule(CompilerInstance &ImportingInstance,
                          StringRef ModuleName,
                          StringRef ModuleFileName,
                          StringRef UmbrellaHeader) {
  // Construct a compiler invocation for creating this module.
  llvm::IntrusiveRefCntPtr<CompilerInvocation> Invocation
    (new CompilerInvocation(ImportingInstance.getInvocation()));
  Invocation->getLangOpts().resetNonModularOptions();
  Invocation->getPreprocessorOpts().resetNonModularOptions();
  Invocation->getPreprocessorOpts().ModuleBuildPath.push_back(ModuleName);
  
  FrontendOptions &FrontendOpts = Invocation->getFrontendOpts();
  FrontendOpts.OutputFile = ModuleFileName.str();
  FrontendOpts.DisableFree = false;
  FrontendOpts.Inputs.clear();
  FrontendOpts.Inputs.push_back(
    std::make_pair(getSourceInputKindFromOptions(Invocation->getLangOpts()), 
                                                 UmbrellaHeader));
  
  Invocation->getDiagnosticOpts().VerifyDiagnostics = 0;
  
  
  assert(ImportingInstance.getInvocation().getModuleHash() ==
           Invocation->getModuleHash() && "Module hash mismatch!");
  
  // Construct a compiler instance that will be used to actually create the
  // module.
  CompilerInstance Instance;
  Instance.setInvocation(&*Invocation);
  Instance.createDiagnostics(/*argc=*/0, /*argv=*/0, 
                             &ImportingInstance.getDiagnosticClient(),
                             /*ShouldOwnClient=*/false);

  // Construct a module-generating action.
  GeneratePCHAction CreateModuleAction(true);
  
  // Execute the action to actually build the module in-place.
  // FIXME: Need to synchronize when multiple processes do this.
  Instance.ExecuteAction(CreateModuleAction);
  
  // Tell the diagnostic client that it's (re-)starting to process a source
  // file.
  // FIXME: This is a hack. We probably want to clone the diagnostic client.
  ImportingInstance.getDiagnosticClient()
    .BeginSourceFile(ImportingInstance.getLangOpts(),
                     &ImportingInstance.getPreprocessor());
} 

ModuleKey CompilerInstance::loadModule(SourceLocation ImportLoc, 
                                       IdentifierInfo &ModuleName,
                                       SourceLocation ModuleNameLoc) {  
  // Determine what file we're searching from.
  SourceManager &SourceMgr = getSourceManager();
  SourceLocation ExpandedImportLoc = SourceMgr.getExpansionLoc(ImportLoc);
  const FileEntry *CurFile
    = SourceMgr.getFileEntryForID(SourceMgr.getFileID(ExpandedImportLoc));
  if (!CurFile)
    CurFile = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID());

  // Search for a module with the given name.
  std::string UmbrellaHeader;
  std::string ModuleFileName;
  const FileEntry *ModuleFile
    = PP->getHeaderSearchInfo().lookupModule(ModuleName.getName(),
                                             &ModuleFileName,
                                             &UmbrellaHeader);
  
  bool BuildingModule = false;
  if (!ModuleFile && !UmbrellaHeader.empty()) {
    // We didn't find the module, but there is an umbrella header that
    // can be used to create the module file. Create a separate compilation
    // module to do so.
    
    // Check whether there is a cycle in the module graph.
    SmallVectorImpl<std::string> &ModuleBuildPath
      = getPreprocessorOpts().ModuleBuildPath;
    SmallVectorImpl<std::string>::iterator Pos 
      = std::find(ModuleBuildPath.begin(), ModuleBuildPath.end(),
                  ModuleName.getName());
    if (Pos != ModuleBuildPath.end()) {
      llvm::SmallString<256> CyclePath;
      for (; Pos != ModuleBuildPath.end(); ++Pos) {
        CyclePath += *Pos;
        CyclePath += " -> ";
      }
      CyclePath += ModuleName.getName();
      
      getDiagnostics().Report(ModuleNameLoc, diag::err_module_cycle)
        << ModuleName.getName() << CyclePath;
      return 0;
    }
    
    BuildingModule = true;
    compileModule(*this, ModuleName.getName(), ModuleFileName, UmbrellaHeader);
    ModuleFile = PP->getHeaderSearchInfo().lookupModule(ModuleName.getName());
  }
  
  if (!ModuleFile) {
    getDiagnostics().Report(ModuleNameLoc, 
                            BuildingModule? diag::err_module_not_built
                                          : diag::err_module_not_found)
      << ModuleName.getName()
      << SourceRange(ImportLoc, ModuleNameLoc);
    return 0;
  }
  
  // If we don't already have an ASTReader, create one now.
  if (!ModuleManager) {
    if (!hasASTContext())
      createASTContext();

    std::string Sysroot = getHeaderSearchOpts().Sysroot;
    const PreprocessorOptions &PPOpts = getPreprocessorOpts();
    ModuleManager = new ASTReader(getPreprocessor(), *Context,
                                  Sysroot.empty() ? "" : Sysroot.c_str(),
                                  PPOpts.DisablePCHValidation, 
                                  PPOpts.DisableStatCache);
    if (hasASTConsumer()) {
      ModuleManager->setDeserializationListener(
        getASTConsumer().GetASTDeserializationListener());
      getASTContext().setASTMutationListener(
        getASTConsumer().GetASTMutationListener());
    }
    llvm::OwningPtr<ExternalASTSource> Source;
    Source.reset(ModuleManager);
    getASTContext().setExternalSource(Source);
    if (hasSema())
      ModuleManager->InitializeSema(getSema());
    if (hasASTConsumer())
      ModuleManager->StartTranslationUnit(&getASTConsumer());
  }
  
  // Try to load the module we found.
  switch (ModuleManager->ReadAST(ModuleFile->getName(),
                                 serialization::MK_Module)) {
  case ASTReader::Success:
    break;

  case ASTReader::IgnorePCH:
    // FIXME: The ASTReader will already have complained, but can we showhorn
    // that diagnostic information into a more useful form?
    return 0;
      
  case ASTReader::Failure:
    // Already complained.
    return 0;
  }
  
  // FIXME: The module file's FileEntry makes a poor key indeed!
  return (ModuleKey)ModuleFile;
}

