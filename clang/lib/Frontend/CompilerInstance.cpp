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
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "clang/Frontend/Utils.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Timer.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
#include "llvm/System/Signals.h"
using namespace clang;

CompilerInstance::CompilerInstance()
  : Invocation(new CompilerInvocation()) {
}

CompilerInstance::~CompilerInstance() {
}

void CompilerInstance::setLLVMContext(llvm::LLVMContext *Value) {
  LLVMContext.reset(Value);
}

void CompilerInstance::setInvocation(CompilerInvocation *Value) {
  Invocation.reset(Value);
}

void CompilerInstance::setDiagnostics(Diagnostic *Value) {
  Diagnostics = Value;
}

void CompilerInstance::setTarget(TargetInfo *Value) {
  Target.reset(Value);
}

void CompilerInstance::setFileManager(FileManager *Value) {
  FileMgr.reset(Value);
}

void CompilerInstance::setSourceManager(SourceManager *Value) {
  SourceMgr.reset(Value);
}

void CompilerInstance::setPreprocessor(Preprocessor *Value) {
  PP.reset(Value);
}

void CompilerInstance::setASTContext(ASTContext *Value) {
  Context.reset(Value);
}

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
  llvm::OwningPtr<llvm::raw_ostream> OS(
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

void CompilerInstance::createDiagnostics(int Argc, const char* const *Argv,
                                         DiagnosticClient *Client) {
  Diagnostics = createDiagnostics(getDiagnosticOpts(), Argc, Argv, Client);
}

llvm::IntrusiveRefCntPtr<Diagnostic> 
CompilerInstance::createDiagnostics(const DiagnosticOptions &Opts,
                                    int Argc, const char* const *Argv,
                                    DiagnosticClient *Client) {
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(new Diagnostic(DiagID));

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  if (Client)
    Diags->setClient(Client);
  else
    Diags->setClient(new TextDiagnosticPrinter(llvm::errs(), Opts));

  // Chain in -verify checker, if requested.
  if (Opts.VerifyDiagnostics)
    Diags->setClient(new VerifyDiagnosticsClient(*Diags, Diags->takeClient()));

  if (!Opts.DumpBuildInformation.empty())
    SetUpBuildDumpLog(Opts, Argc, Argv, *Diags);

  // Configure our handling of diagnostics.
  ProcessWarningOptions(*Diags, Opts);

  return Diags;
}

// File Manager

void CompilerInstance::createFileManager() {
  FileMgr.reset(new FileManager());
}

// Source Manager

void CompilerInstance::createSourceManager(FileManager &FileMgr,
                                           const FileSystemOptions &FSOpts) {
  SourceMgr.reset(new SourceManager(getDiagnostics(), FileMgr, FSOpts));
}

// Preprocessor

void CompilerInstance::createPreprocessor() {
  PP.reset(createPreprocessor(getDiagnostics(), getLangOpts(),
                              getPreprocessorOpts(), getHeaderSearchOpts(),
                              getDependencyOutputOpts(), getTarget(),
                              getFrontendOpts(), getFileSystemOpts(),
                              getSourceManager(), getFileManager()));
}

Preprocessor *
CompilerInstance::createPreprocessor(Diagnostic &Diags,
                                     const LangOptions &LangInfo,
                                     const PreprocessorOptions &PPOpts,
                                     const HeaderSearchOptions &HSOpts,
                                     const DependencyOutputOptions &DepOpts,
                                     const TargetInfo &Target,
                                     const FrontendOptions &FEOpts,
                                     const FileSystemOptions &FSOpts,
                                     SourceManager &SourceMgr,
                                     FileManager &FileMgr) {
  // Create a PTH manager if we are using some form of a token cache.
  PTHManager *PTHMgr = 0;
  if (!PPOpts.TokenCache.empty())
    PTHMgr = PTHManager::Create(PPOpts.TokenCache, FileMgr, FSOpts, Diags);

  // Create the Preprocessor.
  HeaderSearch *HeaderInfo = new HeaderSearch(FileMgr, FSOpts);
  Preprocessor *PP = new Preprocessor(Diags, LangInfo, Target,
                                      SourceMgr, *HeaderInfo, PTHMgr,
                                      /*OwnsHeaderSearch=*/true);

  // Note that this is different then passing PTHMgr to Preprocessor's ctor.
  // That argument is used as the IdentifierInfoLookup argument to
  // IdentifierTable's ctor.
  if (PTHMgr) {
    PTHMgr->setPreprocessor(PP);
    PP->setPTHManager(PTHMgr);
  }

  if (PPOpts.DetailedRecord)
    PP->createPreprocessingRecord();
  
  InitializePreprocessor(*PP, FSOpts, PPOpts, HSOpts, FEOpts);

  // Handle generating dependencies, if requested.
  if (!DepOpts.OutputFile.empty())
    AttachDependencyFileGen(*PP, DepOpts);

  return PP;
}

// ASTContext

void CompilerInstance::createASTContext() {
  Preprocessor &PP = getPreprocessor();
  Context.reset(new ASTContext(getLangOpts(), PP.getSourceManager(),
                               getTarget(), PP.getIdentifierTable(),
                               PP.getSelectorTable(), PP.getBuiltinInfo(),
                               /*size_reserve=*/ 0));
}

// ExternalASTSource

void CompilerInstance::createPCHExternalASTSource(llvm::StringRef Path,
                                                  bool DisablePCHValidation,
                                                 void *DeserializationListener){
  llvm::OwningPtr<ExternalASTSource> Source;
  bool Preamble = getPreprocessorOpts().PrecompiledPreambleBytes.first != 0;
  Source.reset(createPCHExternalASTSource(Path, getHeaderSearchOpts().Sysroot,
                                          DisablePCHValidation,
                                          getPreprocessor(), getASTContext(),
                                          DeserializationListener,
                                          Preamble));
  getASTContext().setExternalSource(Source);
}

ExternalASTSource *
CompilerInstance::createPCHExternalASTSource(llvm::StringRef Path,
                                             const std::string &Sysroot,
                                             bool DisablePCHValidation,
                                             Preprocessor &PP,
                                             ASTContext &Context,
                                             void *DeserializationListener,
                                             bool Preamble) {
  llvm::OwningPtr<ASTReader> Reader;
  Reader.reset(new ASTReader(PP, &Context,
                             Sysroot.empty() ? 0 : Sysroot.c_str(),
                             DisablePCHValidation));

  Reader->setDeserializationListener(
            static_cast<ASTDeserializationListener *>(DeserializationListener));
  switch (Reader->ReadAST(Path,
                          Preamble ? ASTReader::Preamble : ASTReader::PCH)) {
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
  const FileEntry *Entry = PP.getFileManager().getFile(Filename,
                                                       PP.getFileSystemOpts());
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
                                               llvm::raw_ostream &OS) {
  if (EnableCodeCompletion(PP, Filename, Line, Column))
    return 0;

  // Set up the creation routine for code-completion.
  return new PrintingCodeCompleteConsumer(ShowMacros, ShowCodePatterns, 
                                          ShowGlobals, OS);
}

void CompilerInstance::createSema(bool CompleteTranslationUnit,
                                  CodeCompleteConsumer *CompletionConsumer) {
  TheSema.reset(new Sema(getPreprocessor(), getASTContext(), getASTConsumer(),
                         CompleteTranslationUnit, CompletionConsumer));
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
      llvm::sys::Path TempPath(it->TempFilename);
      if (EraseFiles)
        TempPath.eraseFromDisk();
      else {
        std::string Error;
        llvm::sys::Path NewOutFile(it->Filename);
        // If '-working-directory' was passed, the output filename should be
        // relative to that.
        FileManager::FixupRelativePath(NewOutFile, getFileSystemOpts());
        if (TempPath.renamePathOnDisk(NewOutFile, &Error)) {
          getDiagnostics().Report(diag::err_fe_unable_to_rename_temp)
            << it->TempFilename << it->Filename << Error;
          TempPath.eraseFromDisk();
        }
      }
    } else if (!it->Filename.empty() && EraseFiles)
      llvm::sys::Path(it->Filename).eraseFromDisk();
      
  }
  OutputFiles.clear();
}

llvm::raw_fd_ostream *
CompilerInstance::createDefaultOutputFile(bool Binary,
                                          llvm::StringRef InFile,
                                          llvm::StringRef Extension) {
  return createOutputFile(getFrontendOpts().OutputFile, Binary,
                          InFile, Extension);
}

llvm::raw_fd_ostream *
CompilerInstance::createOutputFile(llvm::StringRef OutputPath,
                                   bool Binary,
                                   llvm::StringRef InFile,
                                   llvm::StringRef Extension) {
  std::string Error, OutputPathName, TempPathName;
  llvm::raw_fd_ostream *OS = createOutputFile(OutputPath, Error, Binary,
                                              InFile, Extension,
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
CompilerInstance::createOutputFile(llvm::StringRef OutputPath,
                                   std::string &Error,
                                   bool Binary,
                                   llvm::StringRef InFile,
                                   llvm::StringRef Extension,
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
  
  if (OutFile != "-") {
    llvm::sys::Path OutPath(OutFile);
    // Only create the temporary if we can actually write to OutPath, otherwise
    // we want to fail early.
    if (!OutPath.exists() ||
        (OutPath.isRegularFile() && OutPath.canWrite())) {
      // Create a temporary file.
      llvm::sys::Path TempPath(OutFile);
      if (!TempPath.createTemporaryFileOnDisk())
        TempFile = TempPath.str();
    }
  }

  std::string OSFile = OutFile;
  if (!TempFile.empty())
    OSFile = TempFile;

  llvm::OwningPtr<llvm::raw_fd_ostream> OS(
    new llvm::raw_fd_ostream(OSFile.c_str(), Error,
                             (Binary ? llvm::raw_fd_ostream::F_Binary : 0)));
  if (!Error.empty())
    return 0;

  // Make sure the out stream file gets removed if we crash.
  llvm::sys::RemoveFileOnSignal(llvm::sys::Path(OSFile));

  if (ResultPathName)
    *ResultPathName = OutFile;
  if (TempPathName)
    *TempPathName = TempFile;

  return OS.take();
}

// Initialization Utilities

bool CompilerInstance::InitializeSourceManager(llvm::StringRef InputFile) {
  return InitializeSourceManager(InputFile, getDiagnostics(), getFileManager(),
                                 getFileSystemOpts(),
                                 getSourceManager(), getFrontendOpts());
}

bool CompilerInstance::InitializeSourceManager(llvm::StringRef InputFile,
                                               Diagnostic &Diags,
                                               FileManager &FileMgr,
                                               const FileSystemOptions &FSOpts,
                                               SourceManager &SourceMgr,
                                               const FrontendOptions &Opts) {
  // Figure out where to get and map in the main file.
  if (InputFile != "-") {
    const FileEntry *File = FileMgr.getFile(InputFile, FSOpts);
    if (!File) {
      Diags.Report(diag::err_fe_error_reading) << InputFile;
      return false;
    }
    SourceMgr.createMainFileID(File);
  } else {
    llvm::MemoryBuffer *SB = llvm::MemoryBuffer::getSTDIN();
    if (!SB) {
      Diags.Report(diag::err_fe_error_reading_stdin);
      return false;
    }
    const FileEntry *File = FileMgr.getVirtualFile(SB->getBufferIdentifier(),
                                                   SB->getBufferSize(), 0,
                                                   FSOpts);
    SourceMgr.createMainFileID(File);
    SourceMgr.overrideFileContents(File, SB);
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
  llvm::raw_ostream &OS = llvm::errs();

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

  // Return the appropriate status when verifying diagnostics.
  //
  // FIXME: If we could make getNumErrors() do the right thing, we wouldn't need
  // this.
  if (getDiagnosticOpts().VerifyDiagnostics)
    return !static_cast<VerifyDiagnosticsClient&>(
      getDiagnosticClient()).HadErrors();

  return !getDiagnostics().getNumErrors();
}


