//===--- Tooling.cpp - Running clang standalone tools ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements functions to run clang tools standalone instead
//  of running them as a plugin.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Tooling.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

// For chdir, see the comment in ClangTool::run for more information.
#ifdef _WIN32
#  include <direct.h>
#else
#  include <unistd.h>
#endif

namespace clang {
namespace tooling {

ToolAction::~ToolAction() {}

FrontendActionFactory::~FrontendActionFactory() {}

// FIXME: This file contains structural duplication with other parts of the
// code that sets up a compiler to run tools on it, and we should refactor
// it to be based on the same framework.

/// \brief Builds a clang driver initialized for running clang tools.
static clang::driver::Driver *newDriver(clang::DiagnosticsEngine *Diagnostics,
                                        const char *BinaryName) {
  const std::string DefaultOutputName = "a.out";
  clang::driver::Driver *CompilerDriver = new clang::driver::Driver(
    BinaryName, llvm::sys::getDefaultTargetTriple(),
    DefaultOutputName, *Diagnostics);
  CompilerDriver->setTitle("clang_based_tool");
  return CompilerDriver;
}

/// \brief Retrieves the clang CC1 specific flags out of the compilation's jobs.
///
/// Returns NULL on error.
static const llvm::opt::ArgStringList *getCC1Arguments(
    clang::DiagnosticsEngine *Diagnostics,
    clang::driver::Compilation *Compilation) {
  // We expect to get back exactly one Command job, if we didn't something
  // failed. Extract that job from the Compilation.
  const clang::driver::JobList &Jobs = Compilation->getJobs();
  if (Jobs.size() != 1 || !isa<clang::driver::Command>(*Jobs.begin())) {
    SmallString<256> error_msg;
    llvm::raw_svector_ostream error_stream(error_msg);
    Jobs.Print(error_stream, "; ", true);
    Diagnostics->Report(clang::diag::err_fe_expected_compiler_job)
        << error_stream.str();
    return NULL;
  }

  // The one job we find should be to invoke clang again.
  const clang::driver::Command *Cmd =
      cast<clang::driver::Command>(*Jobs.begin());
  if (StringRef(Cmd->getCreator().getName()) != "clang") {
    Diagnostics->Report(clang::diag::err_fe_expected_clang_command);
    return NULL;
  }

  return &Cmd->getArguments();
}

/// \brief Returns a clang build invocation initialized from the CC1 flags.
static clang::CompilerInvocation *newInvocation(
    clang::DiagnosticsEngine *Diagnostics,
    const llvm::opt::ArgStringList &CC1Args) {
  assert(!CC1Args.empty() && "Must at least contain the program name!");
  clang::CompilerInvocation *Invocation = new clang::CompilerInvocation;
  clang::CompilerInvocation::CreateFromArgs(
      *Invocation, CC1Args.data() + 1, CC1Args.data() + CC1Args.size(),
      *Diagnostics);
  Invocation->getFrontendOpts().DisableFree = false;
  Invocation->getCodeGenOpts().DisableFree = false;
  Invocation->getDependencyOutputOpts() = DependencyOutputOptions();
  return Invocation;
}

bool runToolOnCode(clang::FrontendAction *ToolAction, const Twine &Code,
                   const Twine &FileName) {
  return runToolOnCodeWithArgs(
      ToolAction, Code, std::vector<std::string>(), FileName);
}

static std::vector<std::string>
getSyntaxOnlyToolArgs(const std::vector<std::string> &ExtraArgs,
                      StringRef FileName) {
  std::vector<std::string> Args;
  Args.push_back("clang-tool");
  Args.push_back("-fsyntax-only");
  Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
  Args.push_back(FileName.str());
  return Args;
}

bool runToolOnCodeWithArgs(clang::FrontendAction *ToolAction, const Twine &Code,
                           const std::vector<std::string> &Args,
                           const Twine &FileName) {
  SmallString<16> FileNameStorage;
  StringRef FileNameRef = FileName.toNullTerminatedStringRef(FileNameStorage);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions()));
  ToolInvocation Invocation(getSyntaxOnlyToolArgs(Args, FileNameRef), ToolAction,
                            Files.getPtr());

  SmallString<1024> CodeStorage;
  Invocation.mapVirtualFile(FileNameRef,
                            Code.toNullTerminatedStringRef(CodeStorage));
  return Invocation.run();
}

std::string getAbsolutePath(StringRef File) {
  StringRef RelativePath(File);
  // FIXME: Should '.\\' be accepted on Win32?
  if (RelativePath.startswith("./")) {
    RelativePath = RelativePath.substr(strlen("./"));
  }

  SmallString<1024> AbsolutePath = RelativePath;
  llvm::error_code EC = llvm::sys::fs::make_absolute(AbsolutePath);
  assert(!EC);
  (void)EC;
  llvm::sys::path::native(AbsolutePath);
  return AbsolutePath.str();
}

namespace {

class SingleFrontendActionFactory : public FrontendActionFactory {
  FrontendAction *Action;

public:
  SingleFrontendActionFactory(FrontendAction *Action) : Action(Action) {}

  FrontendAction *create() { return Action; }
};

}

ToolInvocation::ToolInvocation(ArrayRef<std::string> CommandLine,
                               ToolAction *Action, FileManager *Files)
    : CommandLine(CommandLine.vec()),
      Action(Action),
      OwnsAction(false),
      Files(Files),
      DiagConsumer(NULL) {}

ToolInvocation::ToolInvocation(ArrayRef<std::string> CommandLine,
                               FrontendAction *FAction, FileManager *Files)
    : CommandLine(CommandLine.vec()),
      Action(new SingleFrontendActionFactory(FAction)),
      OwnsAction(true),
      Files(Files),
      DiagConsumer(NULL) {}

ToolInvocation::~ToolInvocation() {
  if (OwnsAction)
    delete Action;
}

void ToolInvocation::setDiagnosticConsumer(DiagnosticConsumer *D) {
  DiagConsumer = D;
}

void ToolInvocation::mapVirtualFile(StringRef FilePath, StringRef Content) {
  SmallString<1024> PathStorage;
  llvm::sys::path::native(FilePath, PathStorage);
  MappedFileContents[PathStorage] = Content;
}

bool ToolInvocation::run() {
  std::vector<const char*> Argv;
  for (int I = 0, E = CommandLine.size(); I != E; ++I)
    Argv.push_back(CommandLine[I].c_str());
  const char *const BinaryName = Argv[0];
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter DiagnosticPrinter(
      llvm::errs(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<clang::DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      DiagConsumer ? DiagConsumer : &DiagnosticPrinter, false);

  const OwningPtr<clang::driver::Driver> Driver(
      newDriver(&Diagnostics, BinaryName));
  // Since the input might only be virtual, don't check whether it exists.
  Driver->setCheckInputsExist(false);
  const OwningPtr<clang::driver::Compilation> Compilation(
      Driver->BuildCompilation(llvm::makeArrayRef(Argv)));
  const llvm::opt::ArgStringList *const CC1Args = getCC1Arguments(
      &Diagnostics, Compilation.get());
  if (CC1Args == NULL) {
    return false;
  }
  OwningPtr<clang::CompilerInvocation> Invocation(
      newInvocation(&Diagnostics, *CC1Args));
  for (llvm::StringMap<StringRef>::const_iterator
           It = MappedFileContents.begin(), End = MappedFileContents.end();
       It != End; ++It) {
    // Inject the code as the given file name into the preprocessor options.
    const llvm::MemoryBuffer *Input =
        llvm::MemoryBuffer::getMemBuffer(It->getValue());
    Invocation->getPreprocessorOpts().addRemappedFile(It->getKey(), Input);
  }
  return runInvocation(BinaryName, Compilation.get(), Invocation.take());
}

bool ToolInvocation::runInvocation(
    const char *BinaryName,
    clang::driver::Compilation *Compilation,
    clang::CompilerInvocation *Invocation) {
  // Show the invocation, with -v.
  if (Invocation->getHeaderSearchOpts().Verbose) {
    llvm::errs() << "clang Invocation:\n";
    Compilation->getJobs().Print(llvm::errs(), "\n", true);
    llvm::errs() << "\n";
  }

  return Action->runInvocation(Invocation, Files, DiagConsumer);
}

bool FrontendActionFactory::runInvocation(CompilerInvocation *Invocation,
                                          FileManager *Files,
                                          DiagnosticConsumer *DiagConsumer) {
  // Create a compiler instance to handle the actual work.
  clang::CompilerInstance Compiler;
  Compiler.setInvocation(Invocation);
  Compiler.setFileManager(Files);

  // The FrontendAction can have lifetime requirements for Compiler or its
  // members, and we need to ensure it's deleted earlier than Compiler. So we
  // pass it to an OwningPtr declared after the Compiler variable.
  OwningPtr<FrontendAction> ScopedToolAction(create());

  // Create the compilers actual diagnostics engine.
  Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
  if (!Compiler.hasDiagnostics())
    return false;

  Compiler.createSourceManager(*Files);

  const bool Success = Compiler.ExecuteAction(*ScopedToolAction);

  Files->clearStatCaches();
  return Success;
}

ClangTool::ClangTool(const CompilationDatabase &Compilations,
                     ArrayRef<std::string> SourcePaths)
    : Files(new FileManager(FileSystemOptions())), DiagConsumer(NULL) {
  ArgsAdjusters.push_back(new ClangStripOutputAdjuster());
  ArgsAdjusters.push_back(new ClangSyntaxOnlyAdjuster());
  for (unsigned I = 0, E = SourcePaths.size(); I != E; ++I) {
    SmallString<1024> File(getAbsolutePath(SourcePaths[I]));

    std::vector<CompileCommand> CompileCommandsForFile =
      Compilations.getCompileCommands(File.str());
    if (!CompileCommandsForFile.empty()) {
      for (int I = 0, E = CompileCommandsForFile.size(); I != E; ++I) {
        CompileCommands.push_back(std::make_pair(File.str(),
                                  CompileCommandsForFile[I]));
      }
    } else {
      // FIXME: There are two use cases here: doing a fuzzy
      // "find . -name '*.cc' |xargs tool" match, where as a user I don't care
      // about the .cc files that were not found, and the use case where I
      // specify all files I want to run over explicitly, where this should
      // be an error. We'll want to add an option for this.
      llvm::outs() << "Skipping " << File << ". Command line not found.\n";
    }
  }
}

void ClangTool::setDiagnosticConsumer(DiagnosticConsumer *D) {
  DiagConsumer = D;
}

void ClangTool::mapVirtualFile(StringRef FilePath, StringRef Content) {
  MappedFileContents.push_back(std::make_pair(FilePath, Content));
}

void ClangTool::setArgumentsAdjuster(ArgumentsAdjuster *Adjuster) {
  clearArgumentsAdjusters();
  appendArgumentsAdjuster(Adjuster);
}

void ClangTool::appendArgumentsAdjuster(ArgumentsAdjuster *Adjuster) {
  ArgsAdjusters.push_back(Adjuster);
}

void ClangTool::clearArgumentsAdjusters() {
  for (unsigned I = 0, E = ArgsAdjusters.size(); I != E; ++I)
    delete ArgsAdjusters[I];
  ArgsAdjusters.clear();
}

int ClangTool::run(ToolAction *Action) {
  // Exists solely for the purpose of lookup of the resource path.
  // This just needs to be some symbol in the binary.
  static int StaticSymbol;
  // The driver detects the builtin header path based on the path of the
  // executable.
  // FIXME: On linux, GetMainExecutable is independent of the value of the
  // first argument, thus allowing ClangTool and runToolOnCode to just
  // pass in made-up names here. Make sure this works on other platforms.
  std::string MainExecutable =
      llvm::sys::fs::getMainExecutable("clang_tool", &StaticSymbol);

  bool ProcessingFailed = false;
  for (unsigned I = 0; I < CompileCommands.size(); ++I) {
    std::string File = CompileCommands[I].first;
    // FIXME: chdir is thread hostile; on the other hand, creating the same
    // behavior as chdir is complex: chdir resolves the path once, thus
    // guaranteeing that all subsequent relative path operations work
    // on the same path the original chdir resulted in. This makes a difference
    // for example on network filesystems, where symlinks might be switched
    // during runtime of the tool. Fixing this depends on having a file system
    // abstraction that allows openat() style interactions.
    if (chdir(CompileCommands[I].second.Directory.c_str()))
      llvm::report_fatal_error("Cannot chdir into \"" +
                               CompileCommands[I].second.Directory + "\n!");
    std::vector<std::string> CommandLine = CompileCommands[I].second.CommandLine;
    for (unsigned I = 0, E = ArgsAdjusters.size(); I != E; ++I)
      CommandLine = ArgsAdjusters[I]->Adjust(CommandLine);
    assert(!CommandLine.empty());
    CommandLine[0] = MainExecutable;
    // FIXME: We need a callback mechanism for the tool writer to output a
    // customized message for each file.
    DEBUG({
      llvm::dbgs() << "Processing: " << File << ".\n";
    });
    ToolInvocation Invocation(CommandLine, Action, Files.getPtr());
    Invocation.setDiagnosticConsumer(DiagConsumer);
    for (int I = 0, E = MappedFileContents.size(); I != E; ++I) {
      Invocation.mapVirtualFile(MappedFileContents[I].first,
                                MappedFileContents[I].second);
    }
    if (!Invocation.run()) {
      // FIXME: Diagnostics should be used instead.
      llvm::errs() << "Error while processing " << File << ".\n";
      ProcessingFailed = true;
    }
  }
  return ProcessingFailed ? 1 : 0;
}

namespace {

class ASTBuilderAction : public ToolAction {
  std::vector<ASTUnit *> &ASTs;

public:
  ASTBuilderAction(std::vector<ASTUnit *> &ASTs) : ASTs(ASTs) {}

  bool runInvocation(CompilerInvocation *Invocation, FileManager *Files,
                     DiagnosticConsumer *DiagConsumer) {
    // FIXME: This should use the provided FileManager.
    ASTUnit *AST = ASTUnit::LoadFromCompilerInvocation(
        Invocation, CompilerInstance::createDiagnostics(
                        &Invocation->getDiagnosticOpts(), DiagConsumer,
                        /*ShouldOwnClient=*/false));
    if (!AST)
      return false;

    ASTs.push_back(AST);
    return true;
  }
};

}

int ClangTool::buildASTs(std::vector<ASTUnit *> &ASTs) {
  ASTBuilderAction Action(ASTs);
  return run(&Action);
}

ASTUnit *buildASTFromCode(const Twine &Code, const Twine &FileName) {
  return buildASTFromCodeWithArgs(Code, std::vector<std::string>(), FileName);
}

ASTUnit *buildASTFromCodeWithArgs(const Twine &Code,
                                  const std::vector<std::string> &Args,
                                  const Twine &FileName) {
  SmallString<16> FileNameStorage;
  StringRef FileNameRef = FileName.toNullTerminatedStringRef(FileNameStorage);

  std::vector<ASTUnit *> ASTs;
  ASTBuilderAction Action(ASTs);
  ToolInvocation Invocation(getSyntaxOnlyToolArgs(Args, FileNameRef), &Action, 0);

  SmallString<1024> CodeStorage;
  Invocation.mapVirtualFile(FileNameRef,
                            Code.toNullTerminatedStringRef(CodeStorage));
  if (!Invocation.run())
    return 0;

  assert(ASTs.size() == 1);
  return ASTs[0];
}

} // end namespace tooling
} // end namespace clang
