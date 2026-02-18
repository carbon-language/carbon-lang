// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runner.h"

#include <stdlib.h>
#include <unistd.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/CodeGen/ObjectFilePCHContainerWriter.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "clang/Serialization/PCHContainerOperations.h"
#include "common/check.h"
#include "common/error.h"
#include "common/string_helpers.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "third_party/llvm/clang_cc1.h"
#include "toolchain/base/clang_invocation.h"
#include "toolchain/base/install_paths.h"
#include "toolchain/driver/clang_runtimes.h"
#include "toolchain/driver/runtimes_cache.h"
#include "toolchain/driver/tool_runner_base.h"

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

ClangRunner::ClangRunner(
    const InstallPaths* install_paths,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    llvm::raw_ostream* vlog_stream,
    std::optional<std::filesystem::path> override_clang_path)
    : ToolRunnerBase(install_paths, vlog_stream),
      fs_(std::move(fs)),
      clang_path_(override_clang_path ? *std::move(override_clang_path)
                                      : installation_->clang_path()) {}

// Searches an argument list to a Clang execution to determine the expected
// target string, suitable for use with `llvm::Triple`.
//
// If no explicit target flags are present, this defaults to the default
// LLVM target.
//
// Works to handle the most common flags that modify the expected target as
// well as direct target flags.
//
// Note: this has known fidelity issues if the args include separate-value flags
// (`--flag value` style as opposed to `--flag=value`) where the value might
// match the spelling of one of the target flags. For example, args that include
// an output file spelled `-m32` (so `-o` followed by `-m32`) will be
// misinterpreted by considering the value to itself be a flag. Addressing this
// would add substantial complexity, including likely parsing the entire args
// twice with the Clang driver. Instead, our current plan is to document this
// limitation and encourage the use of flags with joined values
// (`--flag=value`).
static auto ComputeClangTarget(llvm::ArrayRef<llvm::StringRef> args)
    -> std::string {
  std::string target = llvm::sys::getDefaultTargetTriple();
  bool explicit_target = false;
  for (auto [i, arg] : llvm::enumerate(args)) {
    if (llvm::StringRef arg_copy = arg; arg_copy.consume_front("--target=")) {
      target = arg_copy.str();
      explicit_target = true;
    } else if ((arg == "--target" || arg == "-target") &&
               (i + 1) < args.size()) {
      target = args[i + 1].str();
      explicit_target = true;
    } else if (!explicit_target &&
               (arg == "--driver-mode=cl" ||
                ((arg == "--driver-mode" || arg == "-driver-mode") &&
                 (i + 1) < args.size() && args[i + 1] == "cl"))) {
      // The `cl.exe` compatible driver mode should switch the default target to
      // a `...-pc-windows-msvc` target. However, a subsequent explicit target
      // should override this.
      llvm::Triple triple(target);
      triple.setVendor(llvm::Triple::PC);
      triple.setOS(llvm::Triple::Win32);
      triple.setEnvironment(llvm::Triple::MSVC);
      target = triple.str();
    } else if (arg == "-m32") {
      llvm::Triple triple(target);
      if (!triple.isArch32Bit()) {
        target = triple.get32BitArchVariant().str();
      }
    } else if (arg == "-m64") {
      llvm::Triple triple(target);
      if (!triple.isArch64Bit()) {
        target = triple.get64BitArchVariant().str();
      }
    }
  }
  return target;
}

// Tries to detect a a non-linking list of Clang arguments to avoid setting up
// the more complete resource directory needed for linking. False negatives are
// fine here, and we use that to keep things simple.
static auto IsNonLinkCommand(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  return llvm::any_of(args, [](llvm::StringRef arg) {
    // Only check the most common cases as we have to do this for each argument.
    // Everything else is rare and likely not worth the cost of searching for
    // since it's fine to have false negatives.
    return arg == "-c" || arg == "-E" || arg == "-S" ||
           arg == "-fsyntax-only" || arg == "--version" || arg == "--help" ||
           arg == "/?" || arg == "--driver-mode=cpp";
  });
}

auto ClangRunner::RunWithPrebuiltRuntimes(llvm::ArrayRef<llvm::StringRef> args,
                                          Runtimes& prebuilt_runtimes,
                                          bool enable_leaking)
    -> ErrorOr<bool> {
  // Check the args to see if we have a known target-independent command. If so,
  // directly dispatch it to avoid the cost of building the target resource
  // directory.
  // TODO: Maybe handle response file expansion similar to the Clang CLI?
  if (args.empty() || args[0].starts_with("-cc1") || IsNonLinkCommand(args)) {
    return RunWithNoRuntimes(args, enable_leaking);
  }

  std::string target = ComputeClangTarget(args);

  CARBON_ASSIGN_OR_RETURN(std::filesystem::path prebuilt_resource_dir_path,
                          prebuilt_runtimes.Get(Runtimes::ClangResourceDir));
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path libunwind_path,
                          prebuilt_runtimes.Get(Runtimes::LibUnwind));
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path libcxx_path,
                          prebuilt_runtimes.Get(Runtimes::Libcxx));
  return RunInternal(args, target, prebuilt_resource_dir_path.native(),
                     std::move(libunwind_path), std::move(libcxx_path),
                     /*link_runtime_libs=*/true, enable_leaking);
}

auto ClangRunner::Run(llvm::ArrayRef<llvm::StringRef> args,
                      Runtimes::Cache& runtimes_cache,
                      llvm::ThreadPoolInterface& runtimes_build_thread_pool,
                      bool enable_leaking) -> ErrorOr<bool> {
  std::string target = ComputeClangTarget(args);

  // Check the args to see if we have a known target-independent command. If so,
  // directly dispatch it to avoid the cost of building the target resource
  // directory.
  // TODO: Maybe handle response file expansion similar to the Clang CLI?
  if (args.empty() || args[0].starts_with("-cc1") || IsNonLinkCommand(args)) {
    // Note that we do allow linking default libraries here -- we want to learn
    // if a command ever goes through this path and Clang thinks it needs to
    // link a library as the goal here is to correctly detect that this will
    // _not_ happen. Suppressing the linking of default libraries would hide a
    // failure in that case.
    return RunInternal(args, target, /*target_resource_dir_path=*/std::nullopt,
                       /*libunwind_path=*/std::nullopt,
                       /*libcxx_path=*/std::nullopt, /*link_runtime_libs=*/true,
                       enable_leaking);
  }

  Runtimes::Cache::Features features = {.target = target};
  CARBON_ASSIGN_OR_RETURN(Runtimes runtimes, runtimes_cache.Lookup(features));

  // We need to build the Clang resource directory for these runtimes. This
  // requires a temporary directory as well as the destination directory for
  // the build. The temporary directory should only be used during the build,
  // not once we are running Clang with the built runtime.
  CARBON_VLOG("Building target resource dir...\n");
  ClangResourceDirBuilder builder(this, &runtimes_build_thread_pool,
                                  llvm::Triple(features.target), &runtimes);
  ClangArchiveRuntimesBuilder<Runtimes::LibUnwind> lib_unwind_builder(
      this, &runtimes_build_thread_pool, llvm::Triple(features.target),
      &runtimes);
  ClangArchiveRuntimesBuilder<Runtimes::Libcxx> libcxx_builder(
      this, &runtimes_build_thread_pool, llvm::Triple(features.target),
      &runtimes);
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resource_dir_path,
                          std::move(builder).Wait());
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path libunwind_path,
                          std::move(lib_unwind_builder).Wait());
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path libcxx_path,
                          std::move(libcxx_builder).Wait());

  // Note that this function always successfully runs `clang` and returns a bool
  // to indicate whether `clang` itself succeeded, not whether the runner was
  // able to run it. As a consequence, even a `false` here is a non-`Error`
  // return.
  return RunInternal(args, target, resource_dir_path.native(),
                     std::move(libunwind_path), std::move(libcxx_path),
                     /*link_runtime_libs=*/true, enable_leaking);
}

auto ClangRunner::RunWithNoRuntimes(llvm::ArrayRef<llvm::StringRef> args,
                                    bool enable_leaking) -> ErrorOr<bool> {
  std::string target = ComputeClangTarget(args);
  return RunInternal(args, target, /*target_resource_dir_path=*/std::nullopt,
                     /*libunwind_path=*/std::nullopt,
                     /*libcxx_path=*/std::nullopt, /*link_runtime_libs=*/false,
                     enable_leaking);
}

auto ClangRunner::RunInternal(
    llvm::ArrayRef<llvm::StringRef> args, llvm::StringRef target,
    std::optional<llvm::StringRef> target_resource_dir_path,

    std::optional<std::filesystem::path> libunwind_path,
    std::optional<std::filesystem::path> libcxx_path, bool link_runtime_libs,
    bool enable_leaking) -> ErrorOr<bool> {
  llvm::BumpPtrAllocator alloc;

  // Handle special dispatch for CC1 commands as they don't use the driver and
  // we don't synthesize any default arguments there.
  if (!args.empty() && args[0].starts_with("-cc1")) {
    llvm::SmallVector<const char*, 64> cstr_args =
        BuildCStrArgs(clang_path_.native(), args, alloc);
    if (args[0] == "-cc1") {
      CARBON_VLOG("Dispatching `-cc1` command line...");
      int exit_code =
          RunClangCC1(*installation_, fs_, cstr_args, enable_leaking);
      // TODO: Should this be forwarding the full exit code?
      return exit_code == 0;
    }

    // Other CC1-based invocations need to dispatch into the `clang_main`
    // routine to work correctly. This means they're not reliable in a library
    // context but currently there is too much logic to reasonably extract here.
    // This at least allows simple cases (often when directly used on the
    // command line) to work correctly.
    //
    // TODO: Factor the relevant code paths into a library API or move this into
    // the busybox dispatch logic.
    CARBON_VLOG("Calling clang_main for a cc1-based invocation...");
    // cstr_args[0] will be the `clang_path` so we don't need the prepend arg.
    llvm::ToolContext tool_context = {
        .Path = cstr_args[0], .PrependArg = "clang", .NeedsPrependArg = false};
    int exit_code = clang_main(
        cstr_args.size(), const_cast<char**>(cstr_args.data()), tool_context);
    // TODO: Should this be forwarding the full exit code?
    return exit_code == 0;
  }

  // We start with a custom prefix of arguments to establish Carbon's default
  // configuration for invoking Clang. These may not all be needed for all
  // invocations, so we also suppress warnings about any that are ignored.
  llvm::SmallVector<std::string> prefix_args;
  prefix_args.push_back("--start-no-unused-arguments");

  AppendDefaultClangArgs(*installation_, target, prefix_args);

  if (link_runtime_libs) {
    // We don't have a direct way to configure the linker search paths in the
    // Clang driver outside of command line flags, so we inject them here with
    // flags. Note that we only inject these as _search_ paths to allow the
    // normal linking rules to govern whether or not to link a given library. We
    // also build our runtimes exclusively as static archives so we don't need
    // to use command line flags to force static runtime linking to occur.
    if (libunwind_path) {
      prefix_args.push_back(
          llvm::formatv("-L{0}/lib", *std::move(libunwind_path)).str());
    }
    if (libcxx_path) {
      prefix_args.push_back(
          llvm::formatv("-L{0}/lib", std::move(libcxx_path)).str());
    }
  } else {
    // If we are suppressing the linking of default libs, ensure we didn't get a
    // path to add to the link for them, or an override of the resource
    // directory.
    CARBON_CHECK(!target_resource_dir_path);
    CARBON_CHECK(!libunwind_path);
    CARBON_CHECK(!libcxx_path);

    // Now suppress all the default library linking, as we don't expect to have
    // any target runtimes on this code path.
    //
    // TODO: What we actually want here is something more like `-nostdlib++`,
    // `-unwindlib=none`, `-rtlib=none`; however, the last of these doesn't
    // exist in Clang and looks tricky to introduce. This is almost certainly
    // wrong, as it likely suppresses the linking of the _C_ standard library,
    // which isn't one of the Clang runtime libraries we're trying to control
    // here. But the only user of this currently doesn't need to distinguish.
    prefix_args.push_back("-nostdlib");
  }
  prefix_args.push_back("--end-no-unused-arguments");

  // Rebuild the args as C-string args.
  llvm::SmallVector<const char*, 64> cstr_args =
      BuildCStrArgs(clang_path_.native(), prefix_args, args, alloc);

  // Expand any response files in the arguments.
  bool is_clang_cl_mode = clang::driver::IsClangCL(
      clang::driver::getDriverMode(clang_path_.native(), cstr_args));
  if (llvm::Error error = clang::driver::expandResponseFiles(
          cstr_args, is_clang_cl_mode, alloc, fs_.get())) {
    return Error(llvm::toString(std::move(error)));
  }

  CARBON_VLOG("Running Clang driver with the following arguments:\n");
  for (const char* cstr_arg : llvm::ArrayRef(cstr_args)) {
    CARBON_VLOG("    '{0}'\n", cstr_arg);
  }

  // Create the diagnostic options and parse arguments controlling them out of
  // our arguments.
  std::unique_ptr<clang::DiagnosticOptions> diagnostic_options =
      clang::CreateAndPopulateDiagOpts(cstr_args);

  // TODO: We don't yet support serializing diagnostics the way the actual
  // `clang` command line does. Unclear if we need to or not, but it would need
  // a bit more logic here to set up chained consumers.
  clang::TextDiagnosticPrinter diagnostic_client(llvm::errs(),
                                                 *diagnostic_options);

  // Note that the `DiagnosticsEngine` takes ownership (via a ref count) of the
  // DiagnosticIDs, unlike the other parameters.
  clang::DiagnosticsEngine diagnostics(clang::DiagnosticIDs::create(),
                                       *diagnostic_options, &diagnostic_client,
                                       /*ShouldOwnClient=*/false);
  clang::ProcessWarningOptions(diagnostics, *diagnostic_options, *fs_);

  // Note that we configure the driver's *default* target here, not the expected
  // target as that will be parsed out of the command line below.
  clang::driver::Driver driver(clang_path_.native(),
                               llvm::sys::getDefaultTargetTriple(), diagnostics,
                               "clang LLVM compiler", fs_);

  llvm::Triple target_triple(target);

  // We need to set an SDK system root on macOS by default. Setting it here
  // allows a custom sysroot to still be specified on the command line.
  //
  // TODO: A different system root should be used for iOS, watchOS, tvOS.
  // Currently, we're only targeting macOS support though.
  if (target_triple.isMacOSX()) {
    // This is the default CLT system root, shown by `xcrun --show-sdk-path`.
    // We hard code it here to avoid the overhead of subprocessing to `xcrun` on
    // each Clang invocation, but this may need to be updated to search or
    // reflect macOS versions if this changes in the future.
    driver.SysRoot = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk";
  }

  // If we have a target-specific resource directory, set it as the default
  // here, otherwise use the installation's resource directory.
  driver.ResourceDir = target_resource_dir_path
                           ? target_resource_dir_path->str()
                           : installation_->clang_resource_path().native();

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
  auto cc1_main = [this, enable_leaking](
                      llvm::SmallVectorImpl<const char*>& cc1_args) -> int {
    return RunClangCC1(*installation_, fs_, cc1_args, enable_leaking);
  };
  driver.CC1Main = cc1_main;

  std::unique_ptr<clang::driver::Compilation> compilation(
      driver.BuildCompilation(cstr_args));
  CARBON_CHECK(compilation, "Should always successfully allocate!");
  if (compilation->containsError()) {
    // These should have been diagnosed by the driver.
    return false;
  }

  // Make sure our target detection matches Clang's. Sadly, we can't just reuse
  // Clang's as it is available too late.
  // TODO: Use nice diagnostics here rather than a check failure.
  CARBON_CHECK(llvm::Triple(target) == llvm::Triple(driver.getTargetTriple()),
               "Mismatch between the expected target '{0}' and the one "
               "computed by Clang '{1}'",
               target, driver.getTargetTriple());

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
