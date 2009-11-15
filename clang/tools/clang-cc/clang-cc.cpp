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
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/PathDiagnosticClients.h"
#include "clang/Frontend/PreprocessorOptions.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "clang/Frontend/VerifyDiagnosticsClient.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Config/config.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Registry.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Host.h"
#include "llvm/System/Path.h"
#include "llvm/System/Signals.h"
#include "llvm/Target/TargetSelect.h"
#include <cstdlib>
#if HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif

using namespace clang;

//===----------------------------------------------------------------------===//
// Utility Methods
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

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

/// ClangFrontendTimer - The front-end activities should charge time to it with
/// TimeRegion.  The -ftime-report option controls whether this will do
/// anything.
llvm::Timer *ClangFrontendTimer = 0;

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

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  Diagnostic &Diags = *static_cast<Diagnostic*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
}

static TargetInfo *
ConstructCompilerInvocation(CompilerInvocation &Opts, Diagnostic &Diags,
                            const char *Argv0, bool &IsAST) {
  // Initialize frontend options.
  InitializeFrontendOptions(Opts.getFrontendOpts());

  // FIXME: The target information in frontend options should be split out into
  // TargetOptions, and the target options in codegen options should move there
  // as well. Then we could properly initialize in layering order.

  // Initialize base triple.  If a -triple option has been specified, use
  // that triple.  Otherwise, default to the host triple.
  llvm::Triple Triple(Opts.getFrontendOpts().TargetTriple);
  if (Triple.getTriple().empty())
    Triple = llvm::Triple(llvm::sys::getHostTriple());

  // Get information about the target being compiled for.
  TargetInfo *Target = TargetInfo::CreateTargetInfo(Triple.getTriple());
  if (!Target) {
    Diags.Report(diag::err_fe_unknown_triple) << Triple.getTriple().c_str();
    return 0;
  }

  // Set the target ABI if specified.
  if (!Opts.getFrontendOpts().TargetABI.empty() &&
      !Target->setABI(Opts.getFrontendOpts().TargetABI)) {
    Diags.Report(diag::err_fe_unknown_target_abi)
      << Opts.getFrontendOpts().TargetABI;
    return 0;
  }

  // Initialize backend options, which may also be used to key some language
  // options.
  InitializeCodeGenOptions(Opts.getCodeGenOpts(), *Target);

  // Determine the input language, we currently require all files to match.
  FrontendOptions::InputKind IK = Opts.getFrontendOpts().Inputs[0].first;
  for (unsigned i = 1, e = Opts.getFrontendOpts().Inputs.size(); i != e; ++i) {
    if (Opts.getFrontendOpts().Inputs[i].first != IK) {
      llvm::errs() << "error: cannot have multiple input files of distinct "
                   << "language kinds without -x\n";
      return 0;
    }
  }

  // Initialize language options.
  //
  // FIXME: These aren't used during operations on ASTs. Split onto a separate
  // code path to make this obvious.
  IsAST = (IK == FrontendOptions::IK_AST);
  if (!IsAST)
    InitializeLangOptions(Opts.getLangOpts(), IK, *Target,
                          Opts.getCodeGenOpts());

  // Initialize the static analyzer options.
  InitializeAnalyzerOptions(Opts.getAnalyzerOpts());

  // Initialize the dependency output options (-M...).
  InitializeDependencyOutputOptions(Opts.getDependencyOutputOpts());

  // Initialize the header search options.
  InitializeHeaderSearchOptions(Opts.getHeaderSearchOpts(),
                                GetBuiltinIncludePath(Argv0),
                                Opts.getLangOpts());

  // Initialize the other preprocessor options.
  InitializePreprocessorOptions(Opts.getPreprocessorOpts());

  // Initialize the preprocessed output options.
  InitializePreprocessorOutputOptions(Opts.getPreprocessorOutputOpts());

  // Finalize some code generation options which are derived from other places.
  if (Opts.getLangOpts().NoBuiltin)
    Opts.getCodeGenOpts().SimplifyLibCalls = 0;
  if (Opts.getLangOpts().CPlusPlus)
    Opts.getCodeGenOpts().NoCommon = 1;
  Opts.getCodeGenOpts().TimePasses = Opts.getFrontendOpts().ShowTimers;

  return Target;
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc, argv);
  CompilerInstance Clang(&llvm::getGlobalContext(), false);

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

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
  bool IsAST;
  Clang.setTarget(
    ConstructCompilerInvocation(Clang.getInvocation(), Clang.getDiagnostics(),
                                argv[0], IsAST));
  if (!Clang.hasTarget())
    return 1;

  // Validate/process some options
  if (Clang.getHeaderSearchOpts().Verbose)
    llvm::errs() << "clang-cc version " CLANG_VERSION_STRING
                 << " based upon " << PACKAGE_STRING
                 << " hosted on " << llvm::sys::getHostTriple() << "\n";

  if (Clang.getFrontendOpts().ShowTimers)
    ClangFrontendTimer = new llvm::Timer("Clang front-end time");

  // Enforce certain implications.
  if (!Clang.getFrontendOpts().ViewClassInheritance.empty())
    Clang.getFrontendOpts().ProgramAction = frontend::InheritanceView;
  if (!Clang.getFrontendOpts().FixItLocations.empty())
    Clang.getFrontendOpts().ProgramAction = frontend::FixIt;

  for (unsigned i = 0, e = Clang.getFrontendOpts().Inputs.size(); i != e; ++i) {
    const std::string &InFile = Clang.getFrontendOpts().Inputs[i].second;

    // If we aren't using an AST file, setup the file and source managers and
    // the preprocessor.
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

    Act->setCurrentTimer(ClangFrontendTimer);
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

  delete ClangFrontendTimer;

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

