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
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
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
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/DynamicLibrary.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
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
  case RewriteBlocks:          return new RewriteBlocksAction();
  case RewriteMacros:          return new RewriteMacrosAction();
  case RewriteObjC:            return new RewriteObjCAction();
  case RewriteTest:            return new RewriteTestAction();
  case RunAnalysis:            return new AnalysisAction();
  case RunPreprocessorOnly:    return new PreprocessOnlyAction();
  }
}

// FIXME: Define the need for this testing away.
static int cc1_test(Diagnostic &Diags,
                    const char **ArgBegin, const char **ArgEnd,
                    const char *Argv0, void *MainAddr) {
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
  CompilerInvocation::CreateFromArgs(Invocation, ArgBegin, ArgEnd,
                                     Argv0, MainAddr, Diags);

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
                                     Invocation2Args.end(), Argv0, MainAddr,
                                     Diags);

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
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(ArgBegin - ArgEnd, ArgBegin);
  CompilerInstance Clang(&llvm::getGlobalContext(), false);

  // Run clang -cc1 test.
  if (ArgBegin != ArgEnd && llvm::StringRef(ArgBegin[0]) == "-cc1test") {
    TextDiagnosticPrinter DiagClient(llvm::errs(), DiagnosticOptions());
    Diagnostic Diags(&DiagClient);
    return cc1_test(Diags, ArgBegin + 1, ArgEnd, Argv0, MainAddr);
  }

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  TextDiagnosticBuffer DiagsBuffer;
  Diagnostic Diags(&DiagsBuffer);
  CompilerInvocation::CreateFromArgs(Clang.getInvocation(), ArgBegin, ArgEnd,
                                     Argv0, MainAddr, Diags);

  // Honor -help.
  if (Clang.getInvocation().getFrontendOpts().ShowHelp) {
    llvm::OwningPtr<driver::OptTable> Opts(driver::createCC1OptTable());
    Opts->PrintHelp(llvm::outs(), "clang -cc1",
                    "LLVM 'Clang' Compiler: http://clang.llvm.org");
    return 0;
  }

  // Honor -version.
  //
  // FIXME: Use a better -version message?
  if (Clang.getInvocation().getFrontendOpts().ShowVersion) {
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

  // If there were any errors in processing arguments, exit now.
  if (Clang.getDiagnostics().getNumErrors())
    return 1;

  // Create the target instance.
  Clang.setTarget(TargetInfo::CreateTargetInfo(Clang.getDiagnostics(),
                                               Clang.getTargetOpts()));
  if (!Clang.hasTarget())
    return 1;

  // Inform the target of the language options
  //
  // FIXME: We shouldn't need to do this, the target should be immutable once
  // created. This complexity should be lifted elsewhere.
  Clang.getTarget().setForcedLangOptions(Clang.getLangOpts());

  // Validate/process some options
  if (Clang.getHeaderSearchOpts().Verbose)
    llvm::errs() << "clang -cc1 version " CLANG_VERSION_STRING
                 << " based upon " << PACKAGE_STRING
                 << " hosted on " << llvm::sys::getHostTriple() << "\n";

  if (Clang.getFrontendOpts().ShowTimers)
    Clang.createFrontendTimer();

  for (unsigned i = 0, e = Clang.getFrontendOpts().Inputs.size(); i != e; ++i) {
    const std::string &InFile = Clang.getFrontendOpts().Inputs[i].second;

    // If we aren't using an AST file, setup the file and source managers and
    // the preprocessor.
    bool IsAST =
      Clang.getFrontendOpts().Inputs[i].first == FrontendOptions::IK_AST;
    if (!IsAST) {
      if (!i) {
        // Create a file manager object to provide access to and cache the
        // filesystem.
        Clang.createFileManager();

        // Create the source manager.
        Clang.createSourceManager();
      } else {
        // Reset the ID tables if we are reusing the SourceManager.
        Clang.getSourceManager().clearIDTables();
      }

      // Create the preprocessor.
      Clang.createPreprocessor();
    }

    llvm::OwningPtr<FrontendAction> Act(CreateFrontendAction(Clang));
    if (!Act)
      break;

    if (Act->BeginSourceFile(Clang, InFile, IsAST)) {
      Act->Execute();
      Act->EndSourceFile();
    }
  }

  if (Clang.getDiagnosticOpts().ShowCarets)
    if (unsigned NumDiagnostics = Clang.getDiagnostics().getNumDiagnostics())
      fprintf(stderr, "%d diagnostic%s generated.\n", NumDiagnostics,
              (NumDiagnostics == 1 ? "" : "s"));

  if (Clang.getFrontendOpts().ShowStats) {
    Clang.getFileManager().PrintStats();
    fprintf(stderr, "\n");
  }

  // Return the appropriate status when verifying diagnostics.
  //
  // FIXME: If we could make getNumErrors() do the right thing, we wouldn't need
  // this.
  if (Clang.getDiagnosticOpts().VerifyDiagnostics)
    return static_cast<VerifyDiagnosticsClient&>(
      Clang.getDiagnosticClient()).HadErrors();

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();

  return (Clang.getDiagnostics().getNumErrors() != 0);
}
