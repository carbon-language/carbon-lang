//===- ClangSrcLocDump.cpp ------------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/JSON.h"

#include "ASTSrcLocProcessor.h"

using namespace clang::tooling;
using namespace clang;
using namespace llvm;

static cl::list<std::string> IncludeDirectories(
    "I", cl::desc("Include directories to use while compiling"),
    cl::value_desc("directory"), cl::Required, cl::OneOrMore, cl::Prefix);

static cl::opt<bool>
    SkipProcessing("skip-processing",
                   cl::desc("Avoid processing the AST header file"),
                   cl::Required, cl::value_desc("bool"));

static cl::opt<std::string> JsonOutputPath("json-output-path",
                                           cl::desc("json output path"),
                                           cl::Required,
                                           cl::value_desc("path"));

class ASTSrcLocGenerationAction : public clang::ASTFrontendAction {
public:
  ASTSrcLocGenerationAction() : Processor(JsonOutputPath) {}

  void ExecuteAction() override {
    clang::ASTFrontendAction::ExecuteAction();
    if (getCompilerInstance().getDiagnostics().getNumErrors() > 0)
      Processor.generateEmpty();
    else
      Processor.generate();
  }

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef File) override {
    return Processor.createASTConsumer(Compiler, File);
  }

private:
  ASTSrcLocProcessor Processor;
};

static const char Filename[] = "ASTTU.cpp";

int main(int argc, const char **argv) {

  cl::ParseCommandLineOptions(argc, argv);

  if (SkipProcessing) {
    std::error_code EC;
    llvm::raw_fd_ostream JsonOut(JsonOutputPath, EC, llvm::sys::fs::OF_Text);
    if (EC)
      return 1;
    JsonOut << formatv("{0:2}", llvm::json::Value(llvm::json::Object()));
    return 0;
  }

  std::vector<std::string> Args;
  Args.push_back("-cc1");

  llvm::transform(IncludeDirectories, std::back_inserter(Args),
                  [](const std::string &IncDir) { return "-I" + IncDir; });

  Args.push_back("-fsyntax-only");
  Args.push_back(Filename);

  std::vector<const char *> Argv(Args.size(), nullptr);
  llvm::transform(Args, Argv.begin(),
                  [](const std::string &Arg) { return Arg.c_str(); });

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts =
      CreateAndPopulateDiagOpts(Argv);

  // Don't output diagnostics, because common scenarios such as
  // cross-compiling fail with diagnostics.  This is not fatal, but
  // just causes attempts to use the introspection API to return no data.
  TextDiagnosticPrinter DiagnosticPrinter(llvm::nulls(), &*DiagOpts);
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      &DiagnosticPrinter, false);

  auto *OFS = new llvm::vfs::OverlayFileSystem(vfs::getRealFileSystem());

  auto *MemFS = new llvm::vfs::InMemoryFileSystem();
  OFS->pushOverlay(MemFS);
  MemFS->addFile(Filename, 0,
                 MemoryBuffer::getMemBuffer("#include \"clang/AST/AST.h\"\n"));

  auto Files = llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(), OFS);

  auto Driver = std::make_unique<driver::Driver>(
      "clang", llvm::sys::getDefaultTargetTriple(), Diagnostics,
      "ast-api-dump-tool", OFS);

  std::unique_ptr<clang::driver::Compilation> Comp(
      Driver->BuildCompilation(llvm::makeArrayRef(Argv)));
  if (!Comp)
    return 1;

  const auto &Jobs = Comp->getJobs();
  if (Jobs.size() != 1 || !isa<driver::Command>(*Jobs.begin())) {
    SmallString<256> error_msg;
    llvm::raw_svector_ostream error_stream(error_msg);
    Jobs.Print(error_stream, "; ", true);
    return 1;
  }

  const auto &Cmd = cast<driver::Command>(*Jobs.begin());
  const llvm::opt::ArgStringList &CC1Args = Cmd.getArguments();

  auto Invocation = std::make_unique<CompilerInvocation>();
  CompilerInvocation::CreateFromArgs(*Invocation, CC1Args, Diagnostics);

  CompilerInstance Compiler(std::make_shared<clang::PCHContainerOperations>());
  Compiler.setInvocation(std::move(Invocation));

  Compiler.createDiagnostics(&DiagnosticPrinter, false);
  if (!Compiler.hasDiagnostics())
    return 1;

  // Suppress "2 errors generated" or similar messages
  Compiler.getDiagnosticOpts().ShowCarets = false;
  Compiler.createSourceManager(*Files);
  Compiler.setFileManager(Files.get());

  ASTSrcLocGenerationAction ScopedToolAction;
  Compiler.ExecuteAction(ScopedToolAction);

  Files->clearStatCache();

  return 0;
}
