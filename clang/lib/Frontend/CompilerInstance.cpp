//===--- CompilerInstance.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PTHManager.h"
#include "clang/Frontend/ChainedDiagnosticClient.h"
#include "clang/Frontend/PCHReader.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "clang/Frontend/Utils.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
using namespace clang;

CompilerInstance::CompilerInstance(llvm::LLVMContext *_LLVMContext,
                                   bool _OwnsLLVMContext)
  : LLVMContext(_LLVMContext),
    OwnsLLVMContext(_OwnsLLVMContext) {
    }

CompilerInstance::~CompilerInstance() {
  if (OwnsLLVMContext)
    delete LLVMContext;
}

void CompilerInstance::setDiagnostics(Diagnostic *Value) {
  Diagnostics.reset(Value);
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

void CompilerInstance::setASTConsumer(ASTConsumer *Value) {
  Consumer.reset(Value);
}

void CompilerInstance::setCodeCompletionConsumer(CodeCompleteConsumer *Value) {
  CompletionConsumer.reset(Value);
}

// Diagnostics

static void SetUpBuildDumpLog(const DiagnosticOptions &DiagOpts,
                              unsigned argc, char **argv,
                              llvm::OwningPtr<DiagnosticClient> &DiagClient) {
  std::string ErrorInfo;
  llvm::raw_ostream *OS =
    new llvm::raw_fd_ostream(DiagOpts.DumpBuildInformation.c_str(), ErrorInfo);
  if (!ErrorInfo.empty()) {
    // FIXME: Do not fail like this.
    llvm::errs() << "error opening -dump-build-information file '"
                 << DiagOpts.DumpBuildInformation << "', option ignored!\n";
    delete OS;
    return;
  }

  (*OS) << "clang-cc command line arguments: ";
  for (unsigned i = 0; i != argc; ++i)
    (*OS) << argv[i] << ' ';
  (*OS) << '\n';

  // Chain in a diagnostic client which will log the diagnostics.
  DiagnosticClient *Logger =
    new TextDiagnosticPrinter(*OS, DiagOpts, /*OwnsOutputStream=*/true);
  DiagClient.reset(new ChainedDiagnosticClient(DiagClient.take(), Logger));
}

void CompilerInstance::createDiagnostics(int Argc, char **Argv) {
  Diagnostics.reset(createDiagnostics(getDiagnosticOpts(), Argc, Argv));

  if (Diagnostics)
    DiagClient.reset(Diagnostics->getClient());
}

Diagnostic *CompilerInstance::createDiagnostics(const DiagnosticOptions &Opts,
                                                int Argc, char **Argv) {
  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  llvm::OwningPtr<DiagnosticClient> DiagClient(
    new TextDiagnosticPrinter(llvm::errs(), Opts));

  // Chain in -verify checker, if requested.
  if (Opts.VerifyDiagnostics)
    DiagClient.reset(new VerifyDiagnosticsClient(DiagClient.take()));

  if (!Opts.DumpBuildInformation.empty())
    SetUpBuildDumpLog(Opts, Argc, Argv, DiagClient);

  // Configure our handling of diagnostics.
  Diagnostic *Diags = new Diagnostic(DiagClient.take());
  if (ProcessWarningOptions(*Diags, Opts))
    return 0;

  return Diags;
}

// File Manager

void CompilerInstance::createFileManager() {
  FileMgr.reset(new FileManager());
}

// Source Manager

void CompilerInstance::createSourceManager() {
  SourceMgr.reset(new SourceManager());
}

// Preprocessor

void CompilerInstance::createPreprocessor() {
  PP.reset(createPreprocessor(getDiagnostics(), getLangOpts(),
                              getPreprocessorOpts(), getHeaderSearchOpts(),
                              getDependencyOutputOpts(), getTarget(),
                              getSourceManager(), getFileManager()));
}

Preprocessor *
CompilerInstance::createPreprocessor(Diagnostic &Diags,
                                     const LangOptions &LangInfo,
                                     const PreprocessorOptions &PPOpts,
                                     const HeaderSearchOptions &HSOpts,
                                     const DependencyOutputOptions &DepOpts,
                                     const TargetInfo &Target,
                                     SourceManager &SourceMgr,
                                     FileManager &FileMgr) {
  // Create a PTH manager if we are using some form of a token cache.
  PTHManager *PTHMgr = 0;
  if (!PPOpts.getTokenCache().empty())
    PTHMgr = PTHManager::Create(PPOpts.getTokenCache(), Diags);

  // FIXME: Don't fail like this.
  if (Diags.hasErrorOccurred())
    exit(1);

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

  InitializePreprocessor(*PP, PPOpts, HSOpts);

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
                               /*FreeMemory=*/ !getFrontendOpts().DisableFree,
                               /*size_reserve=*/ 0));
}

// ExternalASTSource

void CompilerInstance::createPCHExternalASTSource(llvm::StringRef Path) {
  llvm::OwningPtr<ExternalASTSource> Source;
  Source.reset(createPCHExternalASTSource(Path, getHeaderSearchOpts().Sysroot,
                                          getPreprocessor(), getASTContext()));
  getASTContext().setExternalSource(Source);
}

ExternalASTSource *
CompilerInstance::createPCHExternalASTSource(llvm::StringRef Path,
                                             const std::string &Sysroot,
                                             Preprocessor &PP,
                                             ASTContext &Context) {
  llvm::OwningPtr<PCHReader> Reader;
  Reader.reset(new PCHReader(PP, &Context,
                             Sysroot.empty() ? 0 : Sysroot.c_str()));

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

void CompilerInstance::createCodeCompletionConsumer() {
  const ParsedSourceLocation &Loc = getFrontendOpts().CodeCompletionAt;
  CompletionConsumer.reset(
    createCodeCompletionConsumer(getPreprocessor(),
                                 Loc.FileName, Loc.Line, Loc.Column,
                                 getFrontendOpts().DebugCodeCompletionPrinter,
                                 getFrontendOpts().ShowMacrosInCodeCompletion,
                                 llvm::outs()));
}

CodeCompleteConsumer *
CompilerInstance::createCodeCompletionConsumer(Preprocessor &PP,
                                               const std::string &Filename,
                                               unsigned Line,
                                               unsigned Column,
                                               bool UseDebugPrinter,
                                               bool ShowMacros,
                                               llvm::raw_ostream &OS) {
  // Tell the source manager to chop off the given file at a specific
  // line and column.
  const FileEntry *Entry = PP.getFileManager().getFile(Filename);
  if (!Entry) {
    PP.getDiagnostics().Report(diag::err_fe_invalid_code_complete_file)
      << Filename;
    return 0;
  }

  // Truncate the named file at the given line/column.
  PP.getSourceManager().truncateFileAt(Entry, Line, Column);

  // Set up the creation routine for code-completion.
  if (UseDebugPrinter)
    return new PrintingCodeCompleteConsumer(ShowMacros, OS);
  else
    return new CIndexCodeCompleteConsumer(ShowMacros, OS);
}

// Output Files

void CompilerInstance::addOutputFile(llvm::StringRef Path,
                                     llvm::raw_ostream *OS) {
  assert(OS && "Attempt to add empty stream to output list!");
  OutputFiles.push_back(std::make_pair(Path, OS));
}

void CompilerInstance::ClearOutputFiles(bool EraseFiles) {
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
    // FIXME: Don't fail this way.
    llvm::errs() << "ERROR: " << Error << "\n";
    ::exit(1);
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

  llvm::raw_fd_ostream *OS =
    new llvm::raw_fd_ostream(OutFile.c_str(), Error,
                             (Binary ? llvm::raw_fd_ostream::F_Binary : 0));
  if (!OS)
    return 0;

  if (ResultPathName)
    *ResultPathName = OutFile;

  return OS;
}
