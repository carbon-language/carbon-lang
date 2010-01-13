//===-- cc1_main.cpp - Clang CC1 Driver -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang -cc1 functionality, which implements the
// core compiler functionality along with a number of additional tools for
// demonstration and testing purposes.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/CC1Options.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetSelect.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

void LLVMErrorHandler(void *UserData, const std::string &Message) {
  Diagnostic &Diags = *static_cast<Diagnostic*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
}

static FrontendAction *CreateFrontendAction(CompilerInstance &CI) {
  using namespace clang::frontend;

  switch (CI.getFrontendOpts().ProgramAction) {
  default:
    llvm_unreachable("Invalid program action!");

  case ASTDump:                return new ASTDumpAction();
  case ASTPrint:               return new ASTPrintAction();
  case ASTPrintXML:            return new ASTPrintXMLAction();
  case ASTView:                return new ASTViewAction();
  case DumpRawTokens:          return new DumpRawTokensAction();
  case DumpRecordLayouts:      return new DumpRecordAction();
  case DumpTokens:             return new DumpTokensAction();
  case EmitAssembly:           return new EmitAssemblyAction();
  case EmitBC:                 return new EmitBCAction();
  case EmitHTML:               return new HTMLPrintAction();
  case EmitLLVM:               return new EmitLLVMAction();
  case EmitLLVMOnly:           return new EmitLLVMOnlyAction();
  case FixIt:                  return new FixItAction();
  case GeneratePCH:            return new GeneratePCHAction();
  case GeneratePTH:            return new GeneratePTHAction();
  case InheritanceView:        return new InheritanceViewAction();
  case ParseNoop:              return new ParseOnlyAction();
  case ParsePrintCallbacks:    return new PrintParseAction();
  case ParseSyntaxOnly:        return new SyntaxOnlyAction();

  case PluginAction: {
    if (CI.getFrontendOpts().ActionName == "help") {
      llvm::errs() << "clang -cc1 plugins:\n";
      for (FrontendPluginRegistry::iterator it =
             FrontendPluginRegistry::begin(),
             ie = FrontendPluginRegistry::end();
           it != ie; ++it)
        llvm::errs() << "  " << it->getName() << " - " << it->getDesc() << "\n";
      return 0;
    }

    for (FrontendPluginRegistry::iterator it =
           FrontendPluginRegistry::begin(), ie = FrontendPluginRegistry::end();
         it != ie; ++it) {
      if (it->getName() == CI.getFrontendOpts().ActionName)
        return it->instantiate();
    }

    CI.getDiagnostics().Report(diag::err_fe_invalid_plugin_name)
      << CI.getFrontendOpts().ActionName;
    return 0;
  }

  case PrintDeclContext:       return new DeclContextPrintAction();
  case PrintPreprocessedInput: return new PrintPreprocessedAction();
  case RewriteMacros:          return new RewriteMacrosAction();
  case RewriteObjC:            return new RewriteObjCAction();
  case RewriteTest:            return new RewriteTestAction();
  case RunAnalysis:            return new AnalysisAction();
  case RunPreprocessorOnly:    return new PreprocessOnlyAction();
  }
}

// FIXME: Define the need for this testing away.
static int cc1_test(Diagnostic &Diags,
                    const char **ArgBegin, const char **ArgEnd) {
  using namespace clang::driver;

  llvm::errs() << "cc1 argv:";
  for (const char **i = ArgBegin; i != ArgEnd; ++i)
    llvm::errs() << " \"" << *i << '"';
  llvm::errs() << "\n";

  // Parse the arguments.
  OptTable *Opts = createCC1OptTable();
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList *Args = Opts->ParseArgs(ArgBegin, ArgEnd,
                                       MissingArgIndex, MissingArgCount);

  // Check for missing argument error.
  if (MissingArgCount)
    Diags.Report(clang::diag::err_drv_missing_argument)
      << Args->getArgString(MissingArgIndex) << MissingArgCount;

  // Dump the parsed arguments.
  llvm::errs() << "cc1 parsed options:\n";
  for (ArgList::const_iterator it = Args->begin(), ie = Args->end();
       it != ie; ++it)
    (*it)->dump();

  // Create a compiler invocation.
  llvm::errs() << "cc1 creating invocation.\n";
  CompilerInvocation Invocation;
  CompilerInvocation::CreateFromArgs(Invocation, ArgBegin, ArgEnd, Diags);

  // Convert the invocation back to argument strings.
  std::vector<std::string> InvocationArgs;
  Invocation.toArgs(InvocationArgs);

  // Dump the converted arguments.
  llvm::SmallVector<const char*, 32> Invocation2Args;
  llvm::errs() << "invocation argv :";
  for (unsigned i = 0, e = InvocationArgs.size(); i != e; ++i) {
    Invocation2Args.push_back(InvocationArgs[i].c_str());
    llvm::errs() << " \"" << InvocationArgs[i] << '"';
  }
  llvm::errs() << "\n";

  // Convert those arguments to another invocation, and check that we got the
  // same thing.
  CompilerInvocation Invocation2;
  CompilerInvocation::CreateFromArgs(Invocation2, Invocation2Args.begin(),
                                     Invocation2Args.end(), Diags);

  // FIXME: Implement CompilerInvocation comparison.
  if (true) {
    //llvm::errs() << "warning: Invocations differ!\n";

    std::vector<std::string> Invocation2Args;
    Invocation2.toArgs(Invocation2Args);
    llvm::errs() << "invocation2 argv:";
    for (unsigned i = 0, e = Invocation2Args.size(); i != e; ++i)
      llvm::errs() << " \"" << Invocation2Args[i] << '"';
    llvm::errs() << "\n";
  }

  return 0;
}

int cc1_main(const char **ArgBegin, const char **ArgEnd,
             const char *Argv0, void *MainAddr) {
  CompilerInstance Clang(&llvm::getGlobalContext(), false);

  // Run clang -cc1 test.
  if (ArgBegin != ArgEnd && llvm::StringRef(ArgBegin[0]) == "-cc1test") {
    TextDiagnosticPrinter DiagClient(llvm::errs(), DiagnosticOptions());
    Diagnostic Diags(&DiagClient);
    return cc1_test(Diags, ArgBegin + 1, ArgEnd);
  }

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  TextDiagnosticBuffer DiagsBuffer;
  Diagnostic Diags(&DiagsBuffer);
  CompilerInvocation::CreateFromArgs(Clang.getInvocation(), ArgBegin, ArgEnd,
                                     Diags);

  // Infer the builtin include path if unspecified.
  if (Clang.getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang.getHeaderSearchOpts().ResourceDir.empty())
    Clang.getHeaderSearchOpts().ResourceDir =
      CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Honor -help.
  if (Clang.getFrontendOpts().ShowHelp) {
    llvm::OwningPtr<driver::OptTable> Opts(driver::createCC1OptTable());
    Opts->PrintHelp(llvm::outs(), "clang -cc1",
                    "LLVM 'Clang' Compiler: http://clang.llvm.org");
    return 0;
  }

  // Honor -version.
  //
  // FIXME: Use a better -version message?
  if (Clang.getFrontendOpts().ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return 0;
  }

  // Create the actual diagnostics engine.
  Clang.createDiagnostics(ArgEnd - ArgBegin, const_cast<char**>(ArgBegin));
  if (!Clang.hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::llvm_install_error_handler(LLVMErrorHandler,
                                   static_cast<void*>(&Clang.getDiagnostics()));

  DiagsBuffer.FlushDiagnostics(Clang.getDiagnostics());

  // Load any requested plugins.
  for (unsigned i = 0,
         e = Clang.getFrontendOpts().Plugins.size(); i != e; ++i) {
    const std::string &Path = Clang.getFrontendOpts().Plugins[i];
    std::string Error;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(Path.c_str(), &Error))
      Diags.Report(diag::err_fe_unable_to_load_plugin) << Path << Error;
  }

  // If there were errors in processing arguments, don't do anything else.
  bool Success = false;
  if (!Clang.getDiagnostics().getNumErrors()) {
    // Create and execute the frontend action.
    llvm::OwningPtr<FrontendAction> Act(CreateFrontendAction(Clang));
    if (Act)
      Success = Clang.ExecuteAction(*Act);
  }

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();

  return !Success;
}
