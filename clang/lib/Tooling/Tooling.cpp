//===--- Tooling.cpp - Running clang standalone tools --------------------===//
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "JsonCompileCommandLineDatabase.h"
#include <map>
#include <cstdio>

namespace clang {
namespace tooling {

namespace {

// Checks that the input conforms to the argv[] convention as in
// main().  Namely:
//   - it must contain at least a program path,
//   - argv[0], ..., and argv[argc - 1] mustn't be NULL, and
//   - argv[argc] must be NULL.
void ValidateArgv(int argc, char* argv[]) {
  if (argc < 1) {
    fprintf(stderr, "ERROR: argc is %d.  It must be >= 1.\n", argc);
    abort();
  }

  for (int i = 0; i < argc; ++i) {
    if (argv[i] == NULL) {
      fprintf(stderr, "ERROR: argv[%d] is NULL.\n", i);
      abort();
    }
  }

  if (argv[argc] != NULL) {
    fprintf(stderr, "ERROR: argv[argc] isn't NULL.\n");
    abort();
  }
}

} // end namespace

// FIXME: This file contains structural duplication with other parts of the
// code that sets up a compiler to run tools on it, and we should refactor
// it to be based on the same framework.

static clang::Diagnostic* NewTextDiagnostics() {
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(
      new clang::DiagnosticIDs());
  clang::TextDiagnosticPrinter *DiagClient = new clang::TextDiagnosticPrinter(
      llvm::errs(), clang::DiagnosticOptions());
  return new clang::Diagnostic(DiagIDs, DiagClient);
}

// Exists solely for the purpose of lookup of the main executable.
static int StaticSymbol;

/// \brief Builds a clang driver initialized for running clang tools.
static clang::driver::Driver* NewDriver(clang::Diagnostic* Diagnostics,
                                        const char* BinaryName) {
  // This just needs to be some symbol in the binary.
  void* const SymbolAddr = &StaticSymbol;
  const llvm::sys::Path ExePath =
      llvm::sys::Path::GetMainExecutable(BinaryName, SymbolAddr);

  const std::string DefaultOutputName = "a.out";
  clang::driver::Driver* CompilerDriver = new clang::driver::Driver(
      ExePath.str(), llvm::sys::getHostTriple(),
      DefaultOutputName, false, false, *Diagnostics);
  CompilerDriver->setTitle("clang_based_tool");
  return CompilerDriver;
}

/// \brief Retrieves the clang CC1 specific flags out of the compilation's jobs.
/// Returns NULL on error.
static const clang::driver::ArgStringList* GetCC1Arguments(
    clang::Diagnostic* Diagnostics, clang::driver::Compilation* Compilation) {
  // We expect to get back exactly one Command job, if we didn't something
  // failed. Extract that job from the Compilation.
  const clang::driver::JobList &Jobs = Compilation->getJobs();
  if (Jobs.size() != 1 || !isa<clang::driver::Command>(*Jobs.begin())) {
    llvm::SmallString<256> error_msg;
    llvm::raw_svector_ostream error_stream(error_msg);
    Compilation->PrintJob(error_stream, Compilation->getJobs(), "; ", true);
    Diagnostics->Report(clang::diag::err_fe_expected_compiler_job)
        << error_stream.str();
    return NULL;
  }

  // The one job we find should be to invoke clang again.
  const clang::driver::Command *Cmd =
      cast<clang::driver::Command>(*Jobs.begin());
  if (llvm::StringRef(Cmd->getCreator().getName()) != "clang") {
    Diagnostics->Report(clang::diag::err_fe_expected_clang_command);
    return NULL;
  }

  return &Cmd->getArguments();
}

/// \brief Returns a clang build invocation initialized from the CC1 flags.
static clang::CompilerInvocation* NewInvocation(
    clang::Diagnostic* Diagnostics,
    const clang::driver::ArgStringList& CC1Args) {
  clang::CompilerInvocation* Invocation = new clang::CompilerInvocation;
  clang::CompilerInvocation::CreateFromArgs(
      *Invocation, CC1Args.data(), CC1Args.data() + CC1Args.size(),
      *Diagnostics);
  Invocation->getFrontendOpts().DisableFree = false;
  return Invocation;
}

/// \brief Runs the specified clang tool action and returns whether it executed
/// successfully.
static bool RunInvocation(const char* BinaryName,
                          clang::driver::Compilation* Compilation,
                          clang::CompilerInvocation* Invocation,
                          const clang::driver::ArgStringList& CC1Args,
                          clang::FrontendAction* ToolAction) {
  llvm::OwningPtr<clang::FrontendAction> ScopedToolAction(ToolAction);
  // Show the invocation, with -v.
  if (Invocation->getHeaderSearchOpts().Verbose) {
    llvm::errs() << "clang Invocation:\n";
    Compilation->PrintJob(llvm::errs(), Compilation->getJobs(), "\n", true);
    llvm::errs() << "\n";
  }

  // Create a compiler instance to handle the actual work.
  clang::CompilerInstance Compiler;
  Compiler.setInvocation(Invocation);

  // Create the compilers actual diagnostics engine.
  Compiler.createDiagnostics(CC1Args.size(),
                             const_cast<char**>(CC1Args.data()));
  if (!Compiler.hasDiagnostics())
    return false;

  // Infer the builtin include path if unspecified.
  if (Compiler.getHeaderSearchOpts().UseBuiltinIncludes &&
      Compiler.getHeaderSearchOpts().ResourceDir.empty()) {
    // This just needs to be some symbol in the binary.
    void* const SymbolAddr = &StaticSymbol;
    Compiler.getHeaderSearchOpts().ResourceDir =
        clang::CompilerInvocation::GetResourcesPath(BinaryName, SymbolAddr);
  }

  const bool Success = Compiler.ExecuteAction(*ToolAction);
  return Success;
}

/// \brief Converts a string vector representing a Command line into a C
/// string vector representing the Argv (including the trailing NULL).
std::vector<char*> CommandLineToArgv(const std::vector<std::string>* Command) {
  std::vector<char*> Result(Command->size() + 1);
  for (std::vector<char*>::size_type I = 0; I < Command->size(); ++I) {
    Result[I] = const_cast<char*>((*Command)[I].c_str());
  }
  Result[Command->size()] = NULL;
  return Result;
}

bool RunToolWithFlags(
    clang::FrontendAction* ToolAction, int Args, char* Argv[]) {
  ValidateArgv(Args, Argv);
  const llvm::OwningPtr<clang::Diagnostic> Diagnostics(NewTextDiagnostics());
  const llvm::OwningPtr<clang::driver::Driver> Driver(
      NewDriver(Diagnostics.get(), Argv[0]));
  const llvm::OwningPtr<clang::driver::Compilation> Compilation(
      Driver->BuildCompilation(llvm::ArrayRef<const char*>(Argv, Args)));
  const clang::driver::ArgStringList* const CC1Args = GetCC1Arguments(
      Diagnostics.get(), Compilation.get());
  if (CC1Args == NULL) {
    return false;
  }
  llvm::OwningPtr<clang::CompilerInvocation> Invocation(
      NewInvocation(Diagnostics.get(), *CC1Args));
  return RunInvocation(Argv[0], Compilation.get(), Invocation.take(),
                       *CC1Args, ToolAction);
}

/// \brief Runs 'ToolAction' on the code specified by 'FileContents'.
///
/// \param FileContents A mapping from file name to source code. For each
/// entry a virtual file mapping will be created when running the tool.
bool RunToolWithFlagsOnCode(
    const std::vector<std::string>& CommandLine,
    const std::map<std::string, std::string>& FileContents,
    clang::FrontendAction* ToolAction) {
  const std::vector<char*> Argv = CommandLineToArgv(&CommandLine);
  const char* const BinaryName = Argv[0];

  const llvm::OwningPtr<clang::Diagnostic> Diagnostics(NewTextDiagnostics());
  const llvm::OwningPtr<clang::driver::Driver> Driver(
      NewDriver(Diagnostics.get(), BinaryName));

  // Since the Input is only virtual, don't check whether it exists.
  Driver->setCheckInputsExist(false);

  const llvm::OwningPtr<clang::driver::Compilation> Compilation(
      Driver->BuildCompilation(llvm::ArrayRef<const char*>(&Argv[0],
                                                           Argv.size() - 1)));
  const clang::driver::ArgStringList* const CC1Args = GetCC1Arguments(
      Diagnostics.get(), Compilation.get());
  if (CC1Args == NULL) {
    return false;
  }
  llvm::OwningPtr<clang::CompilerInvocation> Invocation(
      NewInvocation(Diagnostics.get(), *CC1Args));

  for (std::map<std::string, std::string>::const_iterator
           It = FileContents.begin(), End = FileContents.end();
       It != End; ++It) {
    // Inject the code as the given file name into the preprocessor options.
    const llvm::MemoryBuffer* Input =
        llvm::MemoryBuffer::getMemBuffer(It->second.c_str());
    Invocation->getPreprocessorOpts().addRemappedFile(It->first.c_str(), Input);
  }

  return RunInvocation(BinaryName, Compilation.get(),
                       Invocation.take(), *CC1Args, ToolAction);
}

bool RunSyntaxOnlyToolOnCode(
    clang::FrontendAction *ToolAction, llvm::StringRef Code) {
  const char* const FileName = "input.cc";
  const char* const CommandLine[] = {
      "clang-tool", "-fsyntax-only", FileName
  };
  std::map<std::string, std::string> FileContents;
  FileContents[FileName] = Code;
  return RunToolWithFlagsOnCode(
      std::vector<std::string>(
          CommandLine,
          CommandLine + sizeof(CommandLine)/sizeof(CommandLine[0])),
      FileContents, ToolAction);
}

namespace {

// A CompileCommandHandler implementation that finds compile commands for a
// specific input file.
//
// FIXME: Implement early exit when JsonCompileCommandLineParser supports it.
class FindHandler : public clang::tooling::CompileCommandHandler {
 public:
  explicit FindHandler(llvm::StringRef File)
      : FileToMatch(File), FoundMatchingCommand(false) {}

  virtual void EndTranslationUnits() {
    if (!FoundMatchingCommand && ErrorMessage.empty()) {
      ErrorMessage = "ERROR: No matching command found.";
    }
  }

  virtual void EndTranslationUnit() {
    if (File == FileToMatch) {
      FoundMatchingCommand = true;
      MatchingCommand.Directory = Directory;
      MatchingCommand.CommandLine = UnescapeJsonCommandLine(Command);
    }
  }

  virtual void HandleKeyValue(llvm::StringRef Key, llvm::StringRef Value) {
    if (Key == "directory") { Directory = Value; }
    else if (Key == "file") { File = Value; }
    else if (Key == "command") { Command = Value; }
    else {
      ErrorMessage = (llvm::Twine("Unknown key: \"") + Key + "\"").str();
    }
  }

  const llvm::StringRef FileToMatch;
  bool FoundMatchingCommand;
  CompileCommand MatchingCommand;
  std::string ErrorMessage;

  llvm::StringRef Directory;
  llvm::StringRef File;
  llvm::StringRef Command;
};

} // end namespace

CompileCommand FindCompileArgsInJsonDatabase(
    llvm::StringRef FileName, llvm::StringRef JsonDatabase,
    std::string &ErrorMessage) {
  FindHandler find_handler(FileName);
  JsonCompileCommandLineParser parser(JsonDatabase, &find_handler);
  if (!parser.Parse()) {
    ErrorMessage = parser.GetErrorMessage();
    return CompileCommand();
  }
  return find_handler.MatchingCommand;
}

} // end namespace tooling
} // end namespace clang

