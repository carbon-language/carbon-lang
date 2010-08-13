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
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "clang/Frontend/Utils.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Timer.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/System/Program.h"
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

void CompilerInstance::setDiagnosticClient(DiagnosticClient *Value) {
  DiagClient.reset(Value);
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
namespace {
  class BinaryDiagnosticSerializer : public DiagnosticClient {
    llvm::raw_ostream &OS;
    SourceManager *SourceMgr;
  public:
    explicit BinaryDiagnosticSerializer(llvm::raw_ostream &OS)
      : OS(OS), SourceMgr(0) { }

    virtual void HandleDiagnostic(Diagnostic::Level DiagLevel,
                                  const DiagnosticInfo &Info);
  };
}

void BinaryDiagnosticSerializer::HandleDiagnostic(Diagnostic::Level DiagLevel,
                                                  const DiagnosticInfo &Info) {
  StoredDiagnostic(DiagLevel, Info).Serialize(OS);
}

static void SetUpBuildDumpLog(const DiagnosticOptions &DiagOpts,
                              unsigned argc, char **argv,
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
  Diags.setClient(new ChainedDiagnosticClient(Diags.getClient(), Logger));
}

void CompilerInstance::createDiagnostics(int Argc, char **Argv) {
  Diagnostics = createDiagnostics(getDiagnosticOpts(), Argc, Argv);

  if (Diagnostics)
    DiagClient.reset(Diagnostics->getClient());
}

llvm::IntrusiveRefCntPtr<Diagnostic> 
CompilerInstance::createDiagnostics(const DiagnosticOptions &Opts,
                                    int Argc, char **Argv) {
  llvm::IntrusiveRefCntPtr<Diagnostic> Diags(new Diagnostic());

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  llvm::OwningPtr<DiagnosticClient> DiagClient;
  if (Opts.BinaryOutput) {
    if (llvm::sys::Program::ChangeStderrToBinary()) {
      // We weren't able to set standard error to binary, which is a
      // bit of a problem. So, just create a text diagnostic printer
      // to complain about this problem, and pretend that the user
      // didn't try to use binary output.
      DiagClient.reset(new TextDiagnosticPrinter(llvm::errs(), Opts));
      Diags->setClient(DiagClient.take());
      Diags->Report(diag::err_fe_stderr_binary);
      return Diags;
    } else {
      DiagClient.reset(new BinaryDiagnosticSerializer(llvm::errs()));
    }
  } else {
    DiagClient.reset(new TextDiagnosticPrinter(llvm::errs(), Opts));
  }

  // Chain in -verify checker, if requested.
  if (Opts.VerifyDiagnostics)
    DiagClient.reset(new VerifyDiagnosticsClient(*Diags, DiagClient.take()));

  Diags->setClient(DiagClient.take());
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

void CompilerInstance::createSourceManager() {
  SourceMgr.reset(new SourceManager(getDiagnostics()));
}

// Preprocessor

void CompilerInstance::createPreprocessor() {
  PP.reset(createPreprocessor(getDiagnostics(), getLangOpts(),
                              getPreprocessorOpts(), getHeaderSearchOpts(),
                              getDependencyOutputOpts(), getTarget(),
                              getFrontendOpts(), getSourceManager(),
                              getFileManager()));
}

Preprocessor *
CompilerInstance::createPreprocessor(Diagnostic &Diags,
                                     const LangOptions &LangInfo,
                                     const PreprocessorOptions &PPOpts,
                                     const HeaderSearchOptions &HSOpts,
                                     const DependencyOutputOptions &DepOpts,
                                     const TargetInfo &Target,
                                     const FrontendOptions &FEOpts,
                                     SourceManager &SourceMgr,
                                     FileManager &FileMgr) {
  // Create a PTH manager if we are using some form of a token cache.
  PTHManager *PTHMgr = 0;
  if (!PPOpts.TokenCache.empty())
    PTHMgr = PTHManager::Create(PPOpts.TokenCache, Diags);

  // Create the Preprocessor.
  HeaderSearch *HeaderInfo = new HeaderSearch(FileMgr);
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
  
  InitializePreprocessor(*PP, PPOpts, HSOpts, FEOpts);

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
  Source.reset(createPCHExternalASTSource(Path, getHeaderSearchOpts().Sysroot,
                                          DisablePCHValidation,
                                          getPreprocessor(), getASTContext(),
                                          DeserializationListener));
  getASTContext().setExternalSource(Source);
}

ExternalASTSource *
CompilerInstance::createPCHExternalASTSource(llvm::StringRef Path,
                                             const std::string &Sysroot,
                                             bool DisablePCHValidation,
                                             Preprocessor &PP,
                                             ASTContext &Context,
                                             void *DeserializationListener) {
  llvm::OwningPtr<PCHReader> Reader;
  Reader.reset(new PCHReader(PP, &Context,
                             Sysroot.empty() ? 0 : Sysroot.c_str(),
                             DisablePCHValidation));

  Reader->setDeserializationListener(
            static_cast<PCHDeserializationListener *>(DeserializationListener));
  switch (Reader->ReadPCH(Path)) {
  case PCHReader::Success:
    // Set the predefines buffer as suggested by the PCH reader. Typically, the
    // predefines buffer will be empty.
    PP.setPredefines(Reader->getSuggestedPredefines());
    return Reader.take();

  case PCHReader::Failure:
    // Unrecoverable failure: don't even try to process the input file.
    break;

  case PCHReader::IgnorePCH:
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
                                   getFrontendOpts().DebugCodeCompletionPrinter,
                                   getFrontendOpts().ShowMacrosInCodeCompletion,
                             getFrontendOpts().ShowCodePatternsInCodeCompletion,
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
                                               bool UseDebugPrinter,
                                               bool ShowMacros,
                                               bool ShowCodePatterns,
                                               llvm::raw_ostream &OS) {
  if (EnableCodeCompletion(PP, Filename, Line, Column))
    return 0;

  // Set up the creation routine for code-completion.
  if (UseDebugPrinter)
    return new PrintingCodeCompleteConsumer(ShowMacros, ShowCodePatterns, OS);
  else
    return new CIndexCodeCompleteConsumer(ShowMacros, ShowCodePatterns, OS);
}

void CompilerInstance::createSema(bool CompleteTranslationUnit,
                                  CodeCompleteConsumer *CompletionConsumer) {
  TheSema.reset(new Sema(getPreprocessor(), getASTContext(), getASTConsumer(),
                         CompleteTranslationUnit, CompletionConsumer));
}

// Output Files

void CompilerInstance::addOutputFile(llvm::StringRef Path,
                                     llvm::raw_ostream *OS) {
  assert(OS && "Attempt to add empty stream to output list!");
  OutputFiles.push_back(std::make_pair(Path, OS));
}

void CompilerInstance::clearOutputFiles(bool EraseFiles) {
  for (std::list< std::pair<std::string, llvm::raw_ostream*> >::iterator
         it = OutputFiles.begin(), ie = OutputFiles.end(); it != ie; ++it) {
    delete it->second;
    if (EraseFiles && !it->first.empty())
      llvm::sys::Path(it->first).eraseFromDisk();
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
  std::string Error, OutputPathName;
  llvm::raw_fd_ostream *OS = createOutputFile(OutputPath, Error, Binary,
                                              InFile, Extension,
                                              &OutputPathName);
  if (!OS) {
    getDiagnostics().Report(diag::err_fe_unable_to_open_output)
      << OutputPath << Error;
    return 0;
  }

  // Add the output file -- but don't try to remove "-", since this means we are
  // using stdin.
  addOutputFile((OutputPathName != "-") ? OutputPathName : "", OS);

  return OS;
}

llvm::raw_fd_ostream *
CompilerInstance::createOutputFile(llvm::StringRef OutputPath,
                                   std::string &Error,
                                   bool Binary,
                                   llvm::StringRef InFile,
                                   llvm::StringRef Extension,
                                   std::string *ResultPathName) {
  std::string OutFile;
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

  llvm::OwningPtr<llvm::raw_fd_ostream> OS(
    new llvm::raw_fd_ostream(OutFile.c_str(), Error,
                             (Binary ? llvm::raw_fd_ostream::F_Binary : 0)));
  if (!Error.empty())
    return 0;

  if (ResultPathName)
    *ResultPathName = OutFile;

  return OS.take();
}

// Initialization Utilities

bool CompilerInstance::InitializeSourceManager(llvm::StringRef InputFile) {
  return InitializeSourceManager(InputFile, getDiagnostics(), getFileManager(),
                                 getSourceManager(), getFrontendOpts());
}

bool CompilerInstance::InitializeSourceManager(llvm::StringRef InputFile,
                                               Diagnostic &Diags,
                                               FileManager &FileMgr,
                                               SourceManager &SourceMgr,
                                               const FrontendOptions &Opts) {
  // Figure out where to get and map in the main file.
  if (InputFile != "-") {
    const FileEntry *File = FileMgr.getFile(InputFile);
    if (File) SourceMgr.createMainFileID(File, SourceLocation());
    if (SourceMgr.getMainFileID().isInvalid()) {
      Diags.Report(diag::err_fe_error_reading) << InputFile;
      return false;
    }
  } else {
    llvm::MemoryBuffer *SB = llvm::MemoryBuffer::getSTDIN();
    if (SB) SourceMgr.createMainFileIDForMemBuffer(SB);
    if (SourceMgr.getMainFileID().isInvalid()) {
      Diags.Report(diag::err_fe_error_reading_stdin);
      return false;
    }
  }

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
    unsigned NumWarnings = getDiagnostics().getNumWarnings();
    unsigned NumErrors = getDiagnostics().getNumErrors() - 
                               getDiagnostics().getNumErrorsSuppressed();
    
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


