//===-- cc1_main.cpp - Clang CC1 Compiler Frontend ------------------------===//
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

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/CC1Options.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetSelect.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// Main driver
//===----------------------------------------------------------------------===//

static void LLVMErrorHandler(void *UserData, const std::string &Message) {
  Diagnostic &Diags = *static_cast<Diagnostic*>(UserData);

  Diags.Report(diag::err_fe_error_backend) << Message;

  // We cannot recover from llvm errors.
  exit(1);
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
  llvm::OwningPtr<CompilerInstance> Clang(new CompilerInstance());
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  // Run clang -cc1 test.
  if (ArgBegin != ArgEnd && llvm::StringRef(ArgBegin[0]) == "-cc1test") {
    Diagnostic Diags(DiagID, new TextDiagnosticPrinter(llvm::errs(), 
                                                       DiagnosticOptions()));
    return cc1_test(Diags, ArgBegin + 1, ArgEnd);
  }

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllMCAsmInfos();
  llvm::InitializeAllMCCodeGenInfos();
  llvm::InitializeAllMCSubtargetInfos();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  Diagnostic Diags(DiagID, DiagsBuffer);
  CompilerInvocation::CreateFromArgs(Clang->getInvocation(), ArgBegin, ArgEnd,
                                     Diags);

  // Infer the builtin include path if unspecified.
  if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang->getHeaderSearchOpts().ResourceDir.empty())
    Clang->getHeaderSearchOpts().ResourceDir =
      CompilerInvocation::GetResourcesPath(Argv0, MainAddr);

  // Create the actual diagnostics engine.
  Clang->createDiagnostics(ArgEnd - ArgBegin, const_cast<char**>(ArgBegin));
  if (!Clang->hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(LLVMErrorHandler,
                                  static_cast<void*>(&Clang->getDiagnostics()));

  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());

  // Execute the frontend actions.
  bool Success = ExecuteCompilerInvocation(Clang.get());

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());

  // Our error handler depends on the Diagnostics object, which we're
  // potentially about to delete. Uninstall the handler now so that any
  // later errors use the default handling behavior instead.
  llvm::remove_fatal_error_handler();

  // When running with -disable-free, don't do any destruction or shutdown.
  if (Clang->getFrontendOpts().DisableFree) {
    if (llvm::AreStatisticsEnabled() || Clang->getFrontendOpts().ShowStats)
      llvm::PrintStatistics();
    Clang.take();
    return !Success;
  }

  // Managed static deconstruction. Useful for making things like
  // -time-passes usable.
  llvm::llvm_shutdown();

  return !Success;
}
