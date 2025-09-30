// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_CLANG_RUNNER_H_
#define CARBON_TOOLCHAIN_DRIVER_CLANG_RUNNER_H_

#include <filesystem>

#include "clang/Basic/DiagnosticIDs.h"
#include "common/error.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/runtimes_cache.h"
#include "toolchain/driver/tool_runner_base.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// Runs Clang in a similar fashion to invoking it with the provided arguments on
// the command line. We use a textual command line interface to allow easily
// incorporating custom command line flags from user invocations that we don't
// parse, but will pass transparently along to Clang itself.
//
// This doesn't literally use a subprocess to invoke Clang; it instead tries to
// directly use the Clang command line driver library. We also work to simplify
// how that driver operates and invoke it in an opinionated way to get the best
// behavior for our expected use cases in the Carbon driver:
//
// - Minimize canonicalization of file names to try to preserve the paths as
//   users type them.
// - Minimize the use of subprocess invocations which are expensive on some
//   operating systems. To the extent possible, we try to directly invoke the
//   Clang logic within this process.
// - Provide programmatic API to control defaults of Clang. For example, causing
//   verbose output.
//
// Note that this makes the current process behave like running Clang -- it uses
// standard output and standard error, and otherwise can only read and write
// files based on their names described in the arguments. It doesn't provide any
// higher-level abstraction such as streams for inputs or outputs.
class ClangRunner : ToolRunnerBase {
 public:
  // Build a Clang runner that uses the provided `exe_name` and `err_stream`.
  //
  // If `verbose` is passed as true, will enable verbose logging to the
  // `err_stream` both from the runner and Clang itself.
  ClangRunner(const InstallPaths* install_paths,
              Runtimes::Cache* on_demand_runtimes_cache,
              llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
              llvm::raw_ostream* vlog_stream = nullptr,
              bool build_runtimes_on_demand = false);

  // Run Clang with the provided arguments.
  //
  // This works to support all of the Clang commandline, including commands that
  // use target-dependent resources like linking. When it detects such commands,
  // it will either use the provided target resource-dir path, or if building
  // runtimes on demand is enabled it will build the needed resource-dir.
  //
  // Returns an error only if unable to successfully run Clang with the
  // arguments. If able to run Clang, no error is returned a bool indicating
  // whether than Clang invocation succeeded is returned.
  //
  // TODO: Eventually, this will need to accept an abstraction that can
  // represent multiple different pre-built runtimes.
  auto Run(llvm::ArrayRef<llvm::StringRef> args,
           Runtimes* prebuilt_runtimes = nullptr) -> ErrorOr<bool>;

  // Run Clang with the provided arguments and without any target-dependent
  // resources.
  //
  // This method can be used to avoid building target-dependent resources when
  // unnecessary, but not all Clang command lines will work correctly.
  // Specifically, compile-only commands will typically work, while linking will
  // not.
  auto RunTargetIndependentCommand(llvm::ArrayRef<llvm::StringRef> args)
      -> bool;

  // Builds the target-specific resource directory for Clang.
  //
  // There is a resource directory installed along side the Clang binary that
  // contains all the target independent files such as headers. However, for
  // target-specific files like runtimes, we build those on demand here and
  // return the path.
  auto BuildTargetResourceDir(const Runtimes::Cache::Features& features,
                              Runtimes& runtimes,
                              const std::filesystem::path& tmp_path)
      -> ErrorOr<std::filesystem::path>;

  // Enable leaking memory.
  //
  // Clang can avoid deallocating some of its memory to improve compile time.
  // However, this isn't compatible with library-based invocations. When using
  // the runner in a context where memory leaks are acceptable, such as from a
  // command line driver, you can use this to enable that leaking behavior. Note
  // that this will not override _explicit_ `args` in a run invocation that
  // cause leaking, it will merely disable Clang's libraries injecting that
  // behavior.
  auto EnableLeakingMemory() -> void { enable_leaking_ = true; }

 private:
  // Handles building the Clang driver and passing the arguments down to it.
  auto RunInternal(llvm::ArrayRef<llvm::StringRef> args, llvm::StringRef target,
                   std::optional<llvm::StringRef> target_resource_dir_path)
      -> bool;

  // Helper to compile a single file of the CRT runtimes.
  auto BuildCrtFile(llvm::StringRef target, llvm::StringRef src_file,
                    const std::filesystem::path& out_path) -> void;

  // Returns the target-specific source files for the builtins runtime library.
  auto CollectBuiltinsSrcFiles(const llvm::Triple& target_triple)
      -> llvm::SmallVector<llvm::StringRef>;

  // Helper to compile a single file of the compiler builtins runtimes.
  auto BuildBuiltinsFile(llvm::StringRef target, llvm::StringRef src_file,
                         const std::filesystem::path& out_path) -> void;

  // Builds the builtins runtime library into the provided archive file path,
  // using the provided objects path for intermediate object files.
  auto BuildBuiltinsLib(llvm::StringRef target,
                        const llvm::Triple& target_triple,
                        const std::filesystem::path& tmp_path,
                        Filesystem::DirRef lib_dir) -> ErrorOr<Success>;

  Runtimes::Cache* runtimes_cache_;

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs_;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids_;

  std::optional<std::filesystem::path> prebuilt_runtimes_path_;

  bool build_runtimes_on_demand_;
  bool enable_leaking_ = false;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_CLANG_RUNNER_H_
