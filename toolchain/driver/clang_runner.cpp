// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runner.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"

// Defined in:
// https://github.com/llvm/llvm-project/blob/main/clang/tools/driver/driver.cpp
//
// While not in a header, this is the API used by llvm-driver.cpp for
// busyboxing.
//
// NOLINTNEXTLINE(readability-identifier-naming)
auto clang_main(int Argc, char** Argv, const llvm::ToolContext& ToolContext)
    -> int;

namespace Carbon {

ClangRunner::ClangRunner(const InstallPaths* install_paths,
                         llvm::StringRef target,
                         llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                         llvm::raw_ostream* vlog_stream)
    : ToolRunnerBase(install_paths, vlog_stream),
      target_(target),
      fs_(std::move(fs)),
      diagnostic_ids_(new clang::DiagnosticIDs()) {}

auto ClangRunner::Run(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  // TODO: Maybe handle response file expansion similar to the Clang CLI?

  std::string clang_path = installation_->clang_path();

  // Rebuild the args as C-string args.
  llvm::OwningArrayRef<char> cstr_arg_storage;
  llvm::SmallVector<const char*, 64> cstr_args =
      BuildCStrArgs("Clang", clang_path, "-v", args, cstr_arg_storage);

  if (!args.empty() && args[0].starts_with("-cc1")) {
    CARBON_VLOG("Calling clang_main for cc1...");
    // cstr_args[0] will be the `clang_path` so we don't need the prepend arg.
    llvm::ToolContext tool_context = {
        .Path = cstr_args[0], .PrependArg = "clang", .NeedsPrependArg = false};
    int exit_code = clang_main(
        cstr_args.size(), const_cast<char**>(cstr_args.data()), tool_context);
    // TODO: Should this be forwarding the full exit code?
    return exit_code == 0;
  }

  CARBON_VLOG("Preparing Clang driver...\n");

  // Create the diagnostic options and parse arguments controlling them out of
  // our arguments.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options =
      clang::CreateAndPopulateDiagOpts(cstr_args);

  // TODO: We don't yet support serializing diagnostics the way the actual
  // `clang` command line does. Unclear if we need to or not, but it would need
  // a bit more logic here to set up chained consumers.
  clang::TextDiagnosticPrinter diagnostic_client(llvm::errs(),
                                                 diagnostic_options.get());

  clang::DiagnosticsEngine diagnostics(
      diagnostic_ids_, diagnostic_options.get(), &diagnostic_client,
      /*ShouldOwnClient=*/false);
  clang::ProcessWarningOptions(diagnostics, *diagnostic_options, *fs_);

  clang::driver::Driver driver(clang_path, target_, diagnostics,
                               "clang LLVM compiler", fs_);

  // Configure the install directory to find other tools and data files.
  //
  // We directly override the detected directory as we use a synthetic path
  // above. This makes it appear that our binary was in the installed binaries
  // directory, and allows finding tools relative to it.
  driver.Dir = installation_->llvm_install_bin();
  CARBON_VLOG("Setting bin directory to: {0}\n", driver.Dir);

  // When there's only one command being run, this will run it in-process.
  // However, a `clang` invocation may cause multiple `cc1` invocations, which
  // still subprocess. See `InProcess` comment at:
  // https://github.com/llvm/llvm-project/blob/86ce8e4504c06ecc3cc42f002ad4eb05cac10925/clang/lib/Driver/Job.cpp#L411-L413
  //
  // Note the subprocessing will effectively call `clang -cc1`, which turns into
  // `carbon-busybox clang -cc1`, which results in an equivalent `clang_main`
  // call.
  //
  // Also note that we only do `-disable-free` filtering in the in-process
  // execution here, as subprocesses leaking memory won't impact this process.
  auto cc1_main = [enable_leaking = enable_leaking_](
                      llvm::SmallVectorImpl<const char*>& cc1_args) -> int {
    // Clang doesn't expose any option to disable injecting `-disable-free` into
    // the CC1 invocation, or any way to append a flag that undoes it. So to
    // avoid leaks, we need to edit the `cc1_args` here and remove
    // `-disable-free`.
    //
    // TODO: We should see if upstream would be open to some configuration hook
    // to suppress this when it is generating the CC1 flags and use that.
    if (!enable_leaking) {
      llvm::erase(cc1_args, llvm::StringRef("-disable-free"));
    }

    // cc1_args[0] will be the `clang_path` so we don't need the prepend arg.
    llvm::ToolContext tool_context = {
        .Path = cc1_args[0], .PrependArg = "clang", .NeedsPrependArg = false};
    return clang_main(cc1_args.size(), const_cast<char**>(cc1_args.data()),
                      tool_context);
  };
  driver.CC1Main = cc1_main;

  std::unique_ptr<clang::driver::Compilation> compilation(
      driver.BuildCompilation(cstr_args));
  CARBON_CHECK(compilation, "Should always successfully allocate!");
  if (compilation->containsError()) {
    // These should have been diagnosed by the driver.
    return false;
  }

  CARBON_VLOG("Running Clang driver...\n");

  llvm::SmallVector<std::pair<int, const clang::driver::Command*>>
      failing_commands;
  int result = driver.ExecuteCompilation(*compilation, failing_commands);

  // Finish diagnosing any failures before we verbosely log the source of those
  // failures.
  diagnostic_client.finish();

  CARBON_VLOG("Execution result code: {0}\n", result);
  for (const auto& [command_result, failing_command] : failing_commands) {
    CARBON_VLOG("Failing command '{0}' with code '{1}' was:\n",
                failing_command->getExecutable(), command_result);
    if (vlog_stream_) {
      failing_command->Print(*vlog_stream_, "\n\n", /*Quote=*/true);
    }
  }

  // Return whether the command was executed successfully.
  return result == 0 && failing_commands.empty();
}

}  // namespace Carbon
