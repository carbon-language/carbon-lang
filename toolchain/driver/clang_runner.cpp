// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runner.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

#include "clang/Basic/Diagnostic.h"
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
#include "common/filesystem.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/LLVMDriver.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/TargetParser/Host.h"
#include "toolchain/base/runtime_sources.h"

namespace Carbon {

ClangRunner::ClangRunner(const InstallPaths* install_paths,
                         llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                         llvm::raw_ostream* vlog_stream)
    : ToolRunnerBase(install_paths, vlog_stream), fs_(std::move(fs)) {}

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
                                          Runtimes& prebuilt_runtimes)
    -> ErrorOr<bool> {
  // Check the args to see if we have a known target-independent command. If so,
  // directly dispatch it to avoid the cost of building the target resource
  // directory.
  // TODO: Maybe handle response file expansion similar to the Clang CLI?
  if (args.empty() || args[0].starts_with("-cc1") || IsNonLinkCommand(args)) {
    return RunWithNoRuntimes(args);
  }

  std::string target = ComputeClangTarget(args);

  CARBON_ASSIGN_OR_RETURN(std::filesystem::path prebuilt_resource_dir_path,
                          prebuilt_runtimes.Get(Runtimes::ClangResourceDir));
  return RunInternal(args, target, prebuilt_resource_dir_path.native());
}

auto ClangRunner::Run(llvm::ArrayRef<llvm::StringRef> args,
                      Runtimes::Cache& runtimes_cache,
                      llvm::ThreadPoolInterface& runtimes_build_thread_pool)
    -> ErrorOr<bool> {
  // Check the args to see if we have a known target-independent command. If so,
  // directly dispatch it to avoid the cost of building the target resource
  // directory.
  // TODO: Maybe handle response file expansion similar to the Clang CLI?
  if (args.empty() || args[0].starts_with("-cc1") || IsNonLinkCommand(args)) {
    return RunWithNoRuntimes(args);
  }

  std::string target = ComputeClangTarget(args);

  CARBON_VLOG("Building target resource dir...\n");
  Runtimes::Cache::Features features = {.target = target};
  CARBON_ASSIGN_OR_RETURN(Runtimes runtimes, runtimes_cache.Lookup(features));

  // We need to build the Clang resource directory for these runtimes. This
  // requires a temporary directory as well as the destination directory for
  // the build. The temporary directory should only be used during the build,
  // not once we are running Clang with the built runtime.
  std::filesystem::path resource_dir_path;
  {
    // Build the temporary directory and threadpool needed.
    CARBON_ASSIGN_OR_RETURN(Filesystem::RemovingDir tmp_dir,
                            Filesystem::MakeTmpDir());

    CARBON_ASSIGN_OR_RETURN(
        resource_dir_path,
        BuildTargetResourceDir(features, runtimes, tmp_dir.abs_path(),
                               runtimes_build_thread_pool));
  }

  // Note that this function always successfully runs `clang` and returns a bool
  // to indicate whether `clang` itself succeeded, not whether the runner was
  // able to run it. As a consequence, even a `false` here is a non-`Error`
  // return.
  return RunInternal(args, target, resource_dir_path.native());
}

auto ClangRunner::RunWithNoRuntimes(llvm::ArrayRef<llvm::StringRef> args)
    -> bool {
  std::string target = ComputeClangTarget(args);
  return RunInternal(args, target, std::nullopt);
}

auto ClangRunner::BuildTargetResourceDir(
    const Runtimes::Cache::Features& features, Runtimes& runtimes,
    const std::filesystem::path& tmp_path, llvm::ThreadPoolInterface& threads)
    -> ErrorOr<std::filesystem::path> {
  // Disable any leaking of memory while building the target resource dir, and
  // restore the previous setting at the end.
  auto restore_leak_flag = llvm::make_scope_exit(
      [&, orig_flag = enable_leaking_] { enable_leaking_ = orig_flag; });
  enable_leaking_ = false;

  CARBON_ASSIGN_OR_RETURN(auto build_dir,
                          runtimes.Build(Runtimes::ClangResourceDir));
  if (std::holds_alternative<std::filesystem::path>(build_dir)) {
    // Found cached build.
    return std::get<std::filesystem::path>(std::move(build_dir));
  }

  auto builder = std::get<Runtimes::Builder>(std::move(build_dir));
  std::string target = features.target;

  // Symlink the installation's `include` and `share` directories.
  std::filesystem::path install_resource_path =
      installation_->clang_resource_path();
  CARBON_RETURN_IF_ERROR(
      builder.dir().Symlink("include", install_resource_path / "include"));
  CARBON_RETURN_IF_ERROR(
      builder.dir().Symlink("share", install_resource_path / "share"));

  // Create the target's `lib` directory.
  std::filesystem::path lib_path =
      std::filesystem::path("lib") / std::string_view(target);
  CARBON_ASSIGN_OR_RETURN(Filesystem::Dir lib_dir,
                          builder.dir().CreateDirectories(lib_path));

  llvm::Triple target_triple(target);
  if (target_triple.isOSWindows()) {
    return Error("TODO: Windows runtimes are untested and not yet supported.");
  }

  llvm::ThreadPoolTaskGroup task_group(threads);

  // For Linux targets, the system libc (typically glibc) doesn't necessarily
  // provide the CRT begin/end files, and so we need to build them.
  if (target_triple.isOSLinux()) {
    task_group.async(
        [this, target,
         path = builder.path() / lib_path / "clang_rt.crtbegin.o"] {
          BuildCrtFile(target, RuntimeSources::CrtBegin, path);
        });
    task_group.async(
        [this, target, path = builder.path() / lib_path / "clang_rt.crtend.o"] {
          BuildCrtFile(target, RuntimeSources::CrtEnd, path);
        });
  }

  CARBON_RETURN_IF_ERROR(
      BuildBuiltinsLib(target, target_triple, tmp_path, lib_dir, threads));

  // Now wait for all the queued builds to complete before we commit the
  // runtimes into the cache.
  task_group.wait();

  return std::move(builder).Commit();
}

auto ClangRunner::RunCC1(llvm::SmallVectorImpl<const char*>& cc1_args) -> int {
  llvm::BumpPtrAllocator allocator;
  llvm::cl::ExpansionContext expansion_context(
      allocator, llvm::cl::TokenizeGNUCommandLine);
  if (llvm::Error error = expansion_context.expandResponseFiles(cc1_args)) {
    llvm::errs() << toString(std::move(error)) << '\n';
    return 1;
  }
  CARBON_CHECK(cc1_args[1] == llvm::StringRef("-cc1"));

  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids =
      clang::DiagnosticIDs::create();

  // Register the support for object-file-wrapped Clang modules.
  auto pch_ops = std::make_shared<clang::PCHContainerOperations>();
  pch_ops->registerWriter(
      std::make_unique<clang::ObjectFilePCHContainerWriter>());
  pch_ops->registerReader(
      std::make_unique<clang::ObjectFilePCHContainerReader>());

  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  clang::DiagnosticOptions diag_opts;
  clang::TextDiagnosticBuffer diag_buffer;
  clang::DiagnosticsEngine diags(diag_ids, diag_opts, &diag_buffer,
                                 /*ShouldOwnClient=*/false);

  // Setup round-trip remarks for the DiagnosticsEngine used in CreateFromArgs.
  if (llvm::find(cc1_args, llvm::StringRef("-Rround-trip-cc1-args")) !=
      cc1_args.end()) {
    diags.setSeverity(clang::diag::remark_cc1_round_trip_generated,
                      clang::diag::Severity::Remark, {});
  }

  auto invocation = std::make_shared<clang::CompilerInvocation>();
  bool success = clang::CompilerInvocation::CreateFromArgs(
      *invocation, llvm::ArrayRef(cc1_args).slice(1), diags, cc1_args[0]);

  // Heap allocate the compiler instance so that if we disable freeing we can
  // discard the pointer without destroying or deallocating it.
  auto clang_instance = std::make_unique<clang::CompilerInstance>(
      std::move(invocation), std::move(pch_ops));

  // Override the disabling of free when we don't want to leak memory.
  if (!enable_leaking_) {
    clang_instance->getFrontendOpts().DisableFree = false;
    clang_instance->getCodeGenOpts().DisableFree = false;
  }

  if (!clang_instance->getFrontendOpts().TimeTracePath.empty()) {
    llvm::timeTraceProfilerInitialize(
        clang_instance->getFrontendOpts().TimeTraceGranularity, cc1_args[0],
        clang_instance->getFrontendOpts().TimeTraceVerbose);
  }

  // TODO: These options should take priority over the actual compilation.
  // However, their implementation is currently not accessible from a library.
  // We should factor the implementation into a reusable location and then use
  // that here.
  CARBON_CHECK(!clang_instance->getFrontendOpts().PrintSupportedCPUs &&
               !clang_instance->getFrontendOpts().PrintSupportedExtensions &&
               !clang_instance->getFrontendOpts().PrintEnabledExtensions);

  // Infer the builtin include path if unspecified.
  if (clang_instance->getHeaderSearchOpts().UseBuiltinIncludes &&
      clang_instance->getHeaderSearchOpts().ResourceDir.empty()) {
    clang_instance->getHeaderSearchOpts().ResourceDir =
        installation_->clang_resource_path();
  }

  // Create the actual diagnostics engine.
  clang_instance->createDiagnostics(*fs_);
  if (!clang_instance->hasDiagnostics()) {
    return EXIT_FAILURE;
  }

  // Now flush the buffered diagnostics into the Clang instance's diagnostic
  // engine. If we've already hit an error, we can exit early once that's done.
  diag_buffer.FlushDiagnostics(clang_instance->getDiagnostics());
  if (!success) {
    clang_instance->getDiagnosticClient().finish();
    return EXIT_FAILURE;
  }

  // Execute the frontend actions.
  {
    llvm::TimeTraceScope time_scope("ExecuteCompiler");
    bool time_passes = clang_instance->getCodeGenOpts().TimePasses;
    if (time_passes) {
      clang_instance->createFrontendTimer();
    }
    llvm::TimeRegion timer(time_passes ? &clang_instance->getFrontendTimer()
                                       : nullptr);
    success = clang::ExecuteCompilerInvocation(clang_instance.get());
  }

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  std::unique_ptr<llvm::raw_ostream> io_file = llvm::CreateInfoOutputFile();
  if (clang_instance->getCodeGenOpts().TimePassesJson) {
    *io_file << "{\n";
    llvm::TimerGroup::printAllJSONValues(*io_file, "");
    *io_file << "\n}\n";
  } else if (!clang_instance->getCodeGenOpts().TimePassesStatsFile) {
    llvm::TimerGroup::printAll(*io_file);
  }
  llvm::TimerGroup::clearAll();

  if (llvm::timeTraceProfilerEnabled()) {
    // It is possible that the compiler instance doesn't own a file manager here
    // if we're compiling a module unit, since the file manager is owned by the
    // AST when we're compiling a module unit. So the file manager may be
    // invalid here.
    //
    // It should be fine to create file manager here since the file system
    // options are stored in the compiler invocation and we can recreate the VFS
    // from the compiler invocation.
    if (!clang_instance->hasFileManager()) {
      clang_instance->createFileManager(fs_);
    }

    if (auto profiler_output = clang_instance->createOutputFile(
            clang_instance->getFrontendOpts().TimeTracePath, /*Binary=*/false,
            /*RemoveFileOnSignal=*/false,
            /*useTemporary=*/false)) {
      llvm::timeTraceProfilerWrite(*profiler_output);
      profiler_output.reset();
      llvm::timeTraceProfilerCleanup();
      clang_instance->clearOutputFiles(false);
    }
  }

  // When running with -disable-free, don't do any destruction or shutdown.
  if (clang_instance->getFrontendOpts().DisableFree) {
    llvm::BuryPointer(std::move(clang_instance));
  }
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

auto ClangRunner::RunInternal(
    llvm::ArrayRef<llvm::StringRef> args, llvm::StringRef target,
    std::optional<llvm::StringRef> target_resource_dir_path) -> bool {
  std::string clang_path = installation_->clang_path();

  // Rebuild the args as C-string args.
  llvm::OwningArrayRef<char> cstr_arg_storage;
  llvm::SmallVector<const char*, 64> cstr_args =
      BuildCStrArgs("Clang", clang_path, "-v", args, cstr_arg_storage);

  // Handle special dispatch for CC1 commands as they don't use the driver.
  if (!args.empty() && args[0].starts_with("-cc1")) {
    CARBON_VLOG("Calling Clang's CC1...");
    int exit_code = RunCC1(cstr_args);
    // TODO: Should this be forwarding the full exit code?
    return exit_code == 0;
  }

  CARBON_VLOG("Preparing Clang driver...\n");

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
  clang::driver::Driver driver(clang_path, llvm::sys::getDefaultTargetTriple(),
                               diagnostics, "clang LLVM compiler", fs_);

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
  // here.
  if (target_resource_dir_path) {
    driver.ResourceDir = target_resource_dir_path->str();
  }

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
  auto cc1_main = [this](llvm::SmallVectorImpl<const char*>& cc1_args) -> int {
    return RunCC1(cc1_args);
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

auto ClangRunner::BuildCrtFile(llvm::StringRef target, llvm::StringRef src_file,
                               const std::filesystem::path& out_path) -> void {
  std::filesystem::path src_path =
      installation_->llvm_runtime_srcs() / std::string_view(src_file);
  CARBON_VLOG("Building `{0}' from `{1}`...\n", out_path, src_path);

  std::string target_arg = llvm::formatv("--target={0}", target).str();
  CARBON_CHECK(RunWithNoRuntimes({
      "-no-canonical-prefixes",
      target_arg,
      "-DCRT_HAS_INITFINI_ARRAY",
      "-DEH_USE_FRAME_REGISTRY",
      "-O3",
      "-fPIC",
      "-ffreestanding",
      "-std=c11",
      "-w",
      "-c",
      "-o",
      out_path.native(),
      src_path.native(),
  }));
}

auto ClangRunner::CollectBuiltinsSrcFiles(const llvm::Triple& target_triple)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> src_files;
  auto append_src_files =
      [&](auto input_srcs,
          llvm::function_ref<bool(llvm::StringRef)> filter_out = {}) {
        for (llvm::StringRef input_src : input_srcs) {
          if (!input_src.ends_with(".c") && !input_src.ends_with(".S")) {
            // Not a compiled file.
            continue;
          }
          if (filter_out && filter_out(input_src)) {
            // Filtered out.
            continue;
          }

          src_files.push_back(input_src);
        }
      };
  append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsGenericSrcs));
  append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsBf16Srcs));
  if (target_triple.isArch64Bit()) {
    append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsTfSrcs));
  }
  auto filter_out_chkstk = [&](llvm::StringRef src) {
    return !target_triple.isOSWindows() || !src.ends_with("chkstk.S");
  };
  if (target_triple.isAArch64()) {
    append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsAarch64Srcs),
                     filter_out_chkstk);
  } else if (target_triple.isX86()) {
    append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsX86ArchSrcs));
    if (target_triple.isArch64Bit()) {
      append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsX86_64Srcs),
                       filter_out_chkstk);
    } else {
      // TODO: This should be turned into a nice user-facing diagnostic about an
      // unsupported target.
      CARBON_CHECK(
          target_triple.isArch32Bit(),
          "The Carbon toolchain doesn't currently support 16-bit x86.");
      append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsI386Srcs),
                       filter_out_chkstk);
    }
  } else {
    // TODO: This should be turned into a nice user-facing diagnostic about an
    // unsupported target.
    CARBON_FATAL("Target architecture is not supported: {0}",
                 target_triple.str());
  }
  return src_files;
}

auto ClangRunner::BuildBuiltinsFile(llvm::StringRef target,
                                    llvm::StringRef src_file,
                                    const std::filesystem::path& out_path)
    -> void {
  std::filesystem::path src_path =
      installation_->llvm_runtime_srcs() / std::string_view(src_file);
  CARBON_VLOG("Building `{0}' from `{1}`...\n", out_path, src_path);

  std::string target_arg = llvm::formatv("--target={0}", target).str();
  CARBON_CHECK(RunWithNoRuntimes({
      "-no-canonical-prefixes",
      target_arg,
      "-O3",
      "-fPIC",
      "-ffreestanding",
      "-fno-builtin",
      "-fomit-frame-pointer",
      "-fvisibility=hidden",
      "-std=c11",
      "-w",
      "-c",
      "-o",
      out_path.native(),
      src_path.native(),
  }));
}

auto ClangRunner::BuildBuiltinsLib(llvm::StringRef target,
                                   const llvm::Triple& target_triple,
                                   const std::filesystem::path& tmp_path,
                                   Filesystem::DirRef lib_dir,
                                   llvm::ThreadPoolInterface& threads)
    -> ErrorOr<Success> {
  llvm::SmallVector<llvm::StringRef> src_files =
      CollectBuiltinsSrcFiles(target_triple);

  CARBON_ASSIGN_OR_RETURN(Filesystem::Dir tmp_dir,
                          Filesystem::Cwd().OpenDir(tmp_path));

  // `NewArchiveMember` isn't default constructable unfortunately, so we first
  // build the objects using an optional wrapper.
  llvm::SmallVector<std::optional<llvm::NewArchiveMember>> objs;
  objs.resize(src_files.size());
  llvm::ThreadPoolTaskGroup member_group(threads);
  for (auto [src_file, obj] : llvm::zip_equal(src_files, objs)) {
    // Create any subdirectories needed for this file.
    std::filesystem::path src_path = src_file.str();
    if (src_path.has_parent_path()) {
      CARBON_RETURN_IF_ERROR(tmp_dir.CreateDirectories(src_path.parent_path()));
    }

    member_group.async([this, target, src_file, &obj, &tmp_path] {
      std::filesystem::path obj_path = tmp_path / std::string_view(src_file);
      obj_path += ".o";
      BuildBuiltinsFile(target, src_file, obj_path);

      auto obj_result = llvm::NewArchiveMember::getFile(obj_path.native(),
                                                        /*Deterministic=*/true);
      CARBON_CHECK(obj_result, "TODO: Diagnose this: {0}",
                   llvm::fmt_consume(obj_result.takeError()));
      obj = std::move(*obj_result);
    });
  }

  // Now build an archive out of the `.o` files for the builtins. Note that we
  // build this directly into the `lib_dir` as this is expected to be a staging
  // directory and cleaned up on errors.
  std::filesystem::path builtins_a_path = "libclang_rt.builtins.a";
  CARBON_ASSIGN_OR_RETURN(
      Filesystem::WriteFile builtins_a_file,
      lib_dir.OpenWriteOnly(builtins_a_path, Filesystem::CreateAlways));

  // Wait for all the object compiles to complete, and then move the objects out
  // of their optional wrappers to match the API required by the archive writer.
  member_group.wait();
  llvm::SmallVector<llvm::NewArchiveMember> unwrapped_objs;
  unwrapped_objs.reserve(objs.size());
  for (auto& obj : objs) {
    unwrapped_objs.push_back(*std::move(obj));
  }
  objs.clear();

  // Write the actual archive.
  {
    llvm::raw_fd_ostream builtins_a_os = builtins_a_file.WriteStream();
    llvm::Error archive_err = llvm::writeArchiveToStream(
        builtins_a_os, unwrapped_objs, llvm::SymtabWritingMode::NormalSymtab,
        target_triple.isOSDarwin() ? llvm::object::Archive::K_DARWIN
                                   : llvm::object::Archive::K_GNU,
        /*Deterministic=*/true, /*Thin=*/false);
    // The presence of an error is `true`.
    if (archive_err) {
      return Error(llvm::toString(std::move(archive_err)));
    }
  }
  CARBON_RETURN_IF_ERROR(std::move(builtins_a_file).Close());
  return Success();
}

}  // namespace Carbon
