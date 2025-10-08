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
#include "llvm/Support/ThreadPool.h"
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
// This class is thread safe, allowing multiple threads to share a single runner
// and concurrently invoke Clang.
//
// This doesn't literally use a subprocess to invoke Clang; it instead tries to
// directly use the Clang command line driver library. We also work to simplify
// how that driver operates and invoke it in an opinionated way to get the best
// behavior for our expected use cases in the Carbon driver:
//
// - Ensure thread-safe invocation of Clang to enable concurrent usage.
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
//
// TODO: Switch the diagnostic machinery to buffer and do locked output so that
// concurrent invocations of Clang don't intermingle their diagnostic output.
//
// TODO: If support for thread-local overrides of `llvm::errs` and `llvm::outs`
// becomes available upstream, also buffer and synchronize those streams to
// further improve the behavior of concurrent invocations.
class ClangRunner : ToolRunnerBase {
 public:
  // Build a Clang runner that uses the provided installation and filesystem.
  //
  // Optionally accepts a `vlog_stream` to enable verbose logging from Carbon to
  // that stream. The verbose output from Clang goes to stderr regardless.
  ClangRunner(const InstallPaths* install_paths,
              llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
              llvm::raw_ostream* vlog_stream = nullptr);

  // Run Clang with the provided arguments and a runtime cache for on-demand
  // runtime building.
  //
  // This works to support all of the Clang commandline, including commands that
  // use target-dependent resources like linking. When it detects such commands,
  // it will use runtimes from the provided cache. If not available in the
  // cache, it will build the necessary runtimes using the provided thread pool
  // both to use and incorporate into the cache.
  //
  // Returns an error only if unable to successfully run Clang with the
  // arguments. If able to run Clang, no error is returned a bool indicating
  // whether than Clang invocation succeeded is returned.
  auto Run(llvm::ArrayRef<llvm::StringRef> args,
           Runtimes::Cache& runtimes_cache,
           llvm::ThreadPoolInterface& runtimes_build_thread_pool)
      -> ErrorOr<bool>;

  // Run Clang with the provided arguments and prebuilt runtimes.
  //
  // Similar to `Run`, but requires and uses pre-built runtimes rather than a
  // cache or building them on demand.
  auto RunWithPrebuiltRuntimes(llvm::ArrayRef<llvm::StringRef> args,
                               Runtimes& prebuilt_runtimes) -> ErrorOr<bool>;

  // Run Clang with the provided arguments and without any target runtimes.
  //
  // This method can be used to avoid building target-dependent resources when
  // unnecessary, but not all Clang command lines will work correctly.
  // Specifically, compile-only commands will typically work, while linking will
  // not.
  //
  // This function simply returns true or false depending on whether Clang runs
  // successfully, as it should display any needed error messages.
  auto RunWithNoRuntimes(llvm::ArrayRef<llvm::StringRef> args) -> bool;

  // Builds the target-specific resource directory for Clang.
  //
  // There is a resource directory installed along side the Clang binary that
  // contains all the target independent files such as headers. However, for
  // target-specific files like runtimes, we build those on demand here and
  // return the path.
  auto BuildTargetResourceDir(const Runtimes::Cache::Features& features,
                              Runtimes& runtimes,
                              const std::filesystem::path& tmp_path,
                              llvm::ThreadPoolInterface& threads)
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
  // Emulates `cc1_main` but in a way that doesn't assume it is running in the
  // main thread and can more easily fit into library calls to do compiles.
  //
  // TODO: Much of the logic here should be factored out of the CC1
  // implementation in Clang's driver and into a reusable part of its libraries.
  // That should allow reducing the code here to a minimal amount.
  auto RunCC1(llvm::SmallVectorImpl<const char*>& cc1_args) -> int;

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
                        Filesystem::DirRef lib_dir,
                        llvm::ThreadPoolInterface& threads) -> ErrorOr<Success>;

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs_;

  bool enable_leaking_ = false;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_CLANG_RUNNER_H_
