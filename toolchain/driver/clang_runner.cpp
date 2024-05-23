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
#include "common/command_line.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"

namespace Carbon {

ClangRunner::ClangRunner(llvm::StringRef install_path, llvm::StringRef target,
                         llvm::raw_ostream* vlog_stream)
    : install_path_(install_path),
      target_(target),
      vlog_stream_(vlog_stream),
      diagnostic_ids_(new clang::DiagnosticIDs()) {}

auto ClangRunner::Run(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  // TODO: Maybe handle response file expansion similar to the Clang CLI?

  // If we have a verbose logging stream, and that stream is the same as
  // `llvm::errs`, then add the `-v` flag so that the driver also prints verbose
  // information.
  bool inject_v_arg = vlog_stream_ == &llvm::errs();
  std::array<llvm::StringRef, 1> v_arg_storage;
  llvm::ArrayRef<llvm::StringRef> maybe_v_arg;
  if (inject_v_arg) {
    v_arg_storage[0] = "-v";
    maybe_v_arg = v_arg_storage;
  }

  CARBON_CHECK(!args.empty());
  CARBON_VLOG() << "Running Clang driver with arguments: \n";

  // Render the arguments into null-terminated C-strings for use by the Clang
  // driver. Command lines can get quite long in build systems so this tries to
  // minimize the memory allocation overhead.

  // Start with a dummy executable name. We'll manually set the install
  // directory below.
  std::array<llvm::StringRef, 1> exe_arg = {"clang-runner"};
  auto args_range =
      llvm::concat<const llvm::StringRef>(exe_arg, maybe_v_arg, args);
  int total_size = 0;
  for (llvm::StringRef arg : args_range) {
    // Accumulate both the string size and a null terminator byte.
    total_size += arg.size() + 1;
  }

  // Allocate one chunk of storage for the actual C-strings and a vector of
  // pointers into the storage.
  llvm::OwningArrayRef<char> cstr_arg_storage(total_size);
  llvm::SmallVector<const char*, 64> cstr_args;
  cstr_args.reserve(args.size() + inject_v_arg + 1);
  for (ssize_t i = 0; llvm::StringRef arg : args_range) {
    cstr_args.push_back(&cstr_arg_storage[i]);
    memcpy(&cstr_arg_storage[i], arg.data(), arg.size());
    i += arg.size();
    cstr_arg_storage[i] = '\0';
    ++i;
  }
  for (const char* cstr_arg : llvm::ArrayRef(cstr_args)) {
    CARBON_VLOG() << "    '" << cstr_arg << "'\n";
  }

  CARBON_VLOG() << "Preparing Clang driver...\n";

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
  clang::ProcessWarningOptions(diagnostics, *diagnostic_options);

  clang::driver::Driver driver("clang-runner", target_, diagnostics);

  // Configure the install directory to find other tools and data files.
  //
  // We directly override the detected directory as we use a synthetic path
  // above. This makes it appear that our binary was in the installed binaries
  // directory, and allows finding tools relative to it.
  driver.Dir = GetLLVMInstallBinPath();
  CARBON_VLOG() << "Setting bin directory to: " << driver.Dir << "\n";

  // TODO: Directly run in-process rather than using a subprocess. This is both
  // more efficient and makes debugging (much) easier. Needs code like:
  // driver.CC1Main = [](llvm::SmallVectorImpl<const char*>& argv) {};
  std::unique_ptr<clang::driver::Compilation> compilation(
      driver.BuildCompilation(cstr_args));
  CARBON_CHECK(compilation) << "Should always successfully allocate!";
  if (compilation->containsError()) {
    // These should have been diagnosed by the driver.
    return false;
  }

  CARBON_VLOG() << "Running Clang driver...\n";

  llvm::SmallVector<std::pair<int, const clang::driver::Command*>>
      failing_commands;
  int result = driver.ExecuteCompilation(*compilation, failing_commands);

  // Finish diagnosing any failures before we verbosely log the source of those
  // failures.
  diagnostic_client.finish();

  CARBON_VLOG() << "Execution result code: " << result << "\n";
  for (const auto& [command_result, failing_command] : failing_commands) {
    CARBON_VLOG() << "Failing command '" << failing_command->getExecutable()
                  << "' with code '" << command_result << "' was:\n";
    if (vlog_stream_) {
      failing_command->Print(*vlog_stream_, "\n\n", /*Quote=*/true);
    }
  }

  // Return whether the command was executed successfully.
  return result == 0 && failing_commands.empty();
}

// Compute the LLVM install's `bin` path from the Carbon install path.
//
// The Carbon toolchain install tree contains a mini LLVM install tree that it
// redirects Clang and LLVM tools to in order to find data and binary paths. We
// don't want to expose this install to the broader system, and so while it
// mirrors a system LLVM install, it nests within the private install area of
// Carbon itself. See the `BUILD` file for more details about the layout of this
// tree.
//
// TODO: Might be worth factoring out a library specifically for managing the
// Carbon toolchain's installation path and locating relevant files associated
// with it rather than doing this one-off here.
auto ClangRunner::GetLLVMInstallBinPath() const -> std::string {
  llvm::SmallString<256> path(install_path_);
  llvm::sys::path::append(path, llvm::sys::path::Style::posix,
                          "lib/carbon/llvm/bin");
  return path.str().str();
}

}  // namespace Carbon
