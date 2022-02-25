//===--- CreateInvocationFromCommandLine.cpp - CompilerInvocation from Args ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Construct a compiler invocation object for command line driver arguments
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Host.h"
using namespace clang;
using namespace llvm::opt;

std::unique_ptr<CompilerInvocation> clang::createInvocationFromCommandLine(
    ArrayRef<const char *> ArgList, IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS, bool ShouldRecoverOnErorrs,
    std::vector<std::string> *CC1Args) {
  if (!Diags.get()) {
    // No diagnostics engine was provided, so create our own diagnostics object
    // with the default options.
    Diags = CompilerInstance::createDiagnostics(new DiagnosticOptions);
  }

  SmallVector<const char *, 16> Args(ArgList.begin(), ArgList.end());

  // FIXME: Find a cleaner way to force the driver into restricted modes.
  Args.insert(
      llvm::find_if(
          Args, [](const char *Elem) { return llvm::StringRef(Elem) == "--"; }),
      "-fsyntax-only");

  // FIXME: We shouldn't have to pass in the path info.
  driver::Driver TheDriver(Args[0], llvm::sys::getDefaultTargetTriple(), *Diags,
                           "clang LLVM compiler", VFS);

  // Don't check that inputs exist, they may have been remapped.
  TheDriver.setCheckInputsExist(false);

  std::unique_ptr<driver::Compilation> C(TheDriver.BuildCompilation(Args));
  if (!C)
    return nullptr;

  // Just print the cc1 options if -### was present.
  if (C->getArgs().hasArg(driver::options::OPT__HASH_HASH_HASH)) {
    C->getJobs().Print(llvm::errs(), "\n", true);
    return nullptr;
  }

  // We expect to get back exactly one command job, if we didn't something
  // failed. Offload compilation is an exception as it creates multiple jobs. If
  // that's the case, we proceed with the first job. If caller needs a
  // particular job, it should be controlled via options (e.g.
  // --cuda-{host|device}-only for CUDA) passed to the driver.
  const driver::JobList &Jobs = C->getJobs();
  bool OffloadCompilation = false;
  if (Jobs.size() > 1) {
    for (auto &A : C->getActions()){
      // On MacOSX real actions may end up being wrapped in BindArchAction
      if (isa<driver::BindArchAction>(A))
        A = *A->input_begin();
      if (isa<driver::OffloadAction>(A)) {
        OffloadCompilation = true;
        break;
      }
    }
  }

  bool PickFirstOfMany = OffloadCompilation || ShouldRecoverOnErorrs;
  if (Jobs.size() == 0 || (Jobs.size() > 1 && !PickFirstOfMany)) {
    SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    Jobs.Print(OS, "; ", true);
    Diags->Report(diag::err_fe_expected_compiler_job) << OS.str();
    return nullptr;
  }
  auto Cmd = llvm::find_if(Jobs, [](const driver::Command &Cmd) {
    return StringRef(Cmd.getCreator().getName()) == "clang";
  });
  if (Cmd == Jobs.end()) {
    Diags->Report(diag::err_fe_expected_clang_command);
    return nullptr;
  }

  const ArgStringList &CCArgs = Cmd->getArguments();
  if (CC1Args)
    *CC1Args = {CCArgs.begin(), CCArgs.end()};
  auto CI = std::make_unique<CompilerInvocation>();
  if (!CompilerInvocation::CreateFromArgs(*CI, CCArgs, *Diags, Args[0]) &&
      !ShouldRecoverOnErorrs)
    return nullptr;
  return CI;
}
