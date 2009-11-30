//===--- clang.cpp - C-Language Front-end ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This utility may be invoked in the following manner:
//   clang-cc --help                - Output help info.
//   clang-cc [options]             - Read from stdin.
//   clang-cc [options] file        - Read from "file".
//   clang-cc [options] file1 file2 - Read these files.
//
//===----------------------------------------------------------------------===//

#include "Options.h"
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
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetSelect.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

std::string GetBuiltinIncludePath(const char *Argv0) {
  llvm::sys::Path P =
    llvm::sys::Path::GetMainExecutable(Argv0,
                                       (void*)(intptr_t) GetBuiltinIncludePath);

  if (!P.isEmpty()) {
    P.eraseComponent();  // Remove /clang from foo/bin/clang
    P.eraseComponent();  // Remove /bin   from foo/bin

    // Get foo/lib/clang/<version>/include
    P.appendComponent("lib");
    P.appendComponent("clang");
    P.appendComponent(CLANG_VERSION_STRING);
    P.appendComponent("include");
  }

  return P.str();
}

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  Diagnostic &Diags = *static_cast<Diagnostic*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
}

static FrontendAction *CreateFrontendAction(CompilerInstance &CI) {
  using namespace clang::frontend;

  switch (CI.getFrontendOpts().ProgramAction) {
  default:
    llvm::llvm_unreachable("Invalid program action!");

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
      llvm::errs() << "clang-cc plugins:\n";
      for (FrontendPluginRegistry::iterator it =
             FrontendPluginRegistry::begin(),
             ie = FrontendPluginRegistry::end();
           it != ie; ++it)
        llvm::errs() << "  " << it->getName() << " - " << it->getDesc() << "\n";
      exit(1);
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

static bool ConstructCompilerInvocation(CompilerInvocation &Opts,
                                        Diagnostic &Diags, const char *Argv0) {
  // Initialize target options.
  InitializeTargetOptions(Opts.getTargetOpts());

  // Initialize frontend options.
  InitializeFrontendOptions(Opts.getFrontendOpts());

  // Determine the input language, we currently require all files to match.
  FrontendOptions::InputKind IK = Opts.getFrontendOpts().Inputs[0].first;
  for (unsigned i = 1, e = Opts.getFrontendOpts().Inputs.size(); i != e; ++i) {
    if (Opts.getFrontendOpts().Inputs[i].first != IK) {
      llvm::errs() << "error: cannot have multiple input files of distinct "
                   << "language kinds without -x\n";
      return false;
    }
  }

  // Initialize language options.
  //
  // FIXME: These aren't used during operations on ASTs. Split onto a separate
  // code path to make this obvious.
  if (IK != FrontendOptions::IK_AST)
    InitializeLangOptions(Opts.getLangOpts(), IK);

  // Initialize the static analyzer options.
  InitializeAnalyzerOptions(Opts.getAnalyzerOpts());

  // Initialize the dependency output options (-M...).
  InitializeDependencyOutputOptions(Opts.getDependencyOutputOpts());

  // Initialize the header search options.
  InitializeHeaderSearchOptions(Opts.getHeaderSearchOpts(),
                                GetBuiltinIncludePath(Argv0));

  // Initialize the other preprocessor options.
  InitializePreprocessorOptions(Opts.getPreprocessorOpts());

  // Initialize the preprocessed output options.
  InitializePreprocessorOutputOptions(Opts.getPreprocessorOutputOpts());

  // Initialize backend options.
  InitializeCodeGenOptions(Opts.getCodeGenOpts(), Opts.getLangOpts());

  return true;
}

static int cc1_main(Diagnostic &Diags,
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

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  CompilerInstance Clang(&llvm::getGlobalContext(), false);

  // Run clang -cc1 test.
  if (argc > 1 && llvm::StringRef(argv[1]) == "-cc1") {
    TextDiagnosticPrinter DiagClient(llvm::errs(), DiagnosticOptions());
    Diagnostic Diags(&DiagClient);
    return cc1_main(Diags, (const char**) argv + 2, (const char**) argv + argc,
                    argv[0], (void*) (intptr_t) GetBuiltinIncludePath);
  }

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

#if 1
  llvm::cl::ParseCommandLineOptions(argc, argv,
                              "LLVM 'Clang' Compiler: http://clang.llvm.org\n");

  // Construct the diagnostic engine first, so that we can build a diagnostic
  // client to use for any errors during option handling.
  InitializeDiagnosticOptions(Clang.getDiagnosticOpts());
  Clang.createDiagnostics(argc, argv);
  if (!Clang.hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::llvm_install_error_handler(LLVMErrorHandler,
                                   static_cast<void*>(&Clang.getDiagnostics()));

  // Now that we have initialized the diagnostics engine, create the target and
  // the compiler invocation object.
  //
  // FIXME: We should move .ast inputs to taking a separate path, they are
  // really quite different.
  if (!ConstructCompilerInvocation(Clang.getInvocation(),
                                   Clang.getDiagnostics(), argv[0]))
    return 1;
#else
  // Buffer diagnostics from argument parsing.
  TextDiagnosticBuffer DiagsBuffer;
  Diagnostic Diags(&DiagsBuffer);

  CompilerInvocation::CreateFromArgs(Clang.getInvocation(),
                                     (const char**) argv + 1,
                                     (const char**) argv + argc, argv[0],
                                     (void*)(intptr_t) GetBuiltinIncludePath,
                                     Diags);

  // Create the actual diagnostics engine.
  Clang.createDiagnostics(argc, argv);
  if (!Clang.hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::llvm_install_error_handler(LLVMErrorHandler,
                                   static_cast<void*>(&Clang.getDiagnostics()));

  DiagsBuffer.FlushDiagnostics(Clang.getDiagnostics());

  // If there were any errors in processing arguments, exit now.
  if (Clang.getDiagnostics().getNumErrors())
    return 1;
#endif

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
    llvm::errs() << "clang-cc version " CLANG_VERSION_STRING
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

