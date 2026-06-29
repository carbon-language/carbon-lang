// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>

#include "common/command_line.h"
#include "common/pretty_stack_trace_function.h"
#include "common/version.h"
#include "toolchain/driver/build_runtimes_subcommand.h"
#include "toolchain/driver/build_subcommand.h"
#include "toolchain/driver/clang_subcommand.h"
#include "toolchain/driver/compile_subcommand.h"
#include "toolchain/driver/config_subcommand.h"
#include "toolchain/driver/format_subcommand.h"
#include "toolchain/driver/language_server_subcommand.h"
#include "toolchain/driver/link_subcommand.h"
#include "toolchain/driver/lld_subcommand.h"
#include "toolchain/driver/llvm_subcommand.h"

namespace Carbon {

namespace {
struct Options {
  static const CommandLine::CommandInfo Info;

  auto Build(CommandLine::CommandBuilder& b) -> void;

  bool verbose = false;
  bool fuzzing = false;
  bool include_diagnostic_kind = false;
  bool threads = true;
  bool build_runtimes_on_demand = false;

  llvm::StringRef runtimes_cache_path;
  llvm::StringRef prebuilt_runtimes_path;

  BuildRuntimesSubcommand runtimes;
  BuildSubcommand build;
  ClangSubcommand clang;
  CompileSubcommand compile;
  ConfigSubcommand config;
  FormatSubcommand format;
  LanguageServerSubcommand language_server;
  LinkSubcommand link;
  LldSubcommand lld;
  LLVMSubcommand llvm;

  // On success, this is set to the subcommand to run.
  DriverSubcommand* selected_subcommand = nullptr;
};
}  // namespace

// Note that this is not constexpr so that it can include information generated
// in separate translation units and potentially overridden at link time in the
// version string.
const CommandLine::CommandInfo Options::Info = {
    .name = "carbon",
    .version = Version::ToolchainInfo,
    .help = R"""(
This is the unified Carbon Language toolchain driver. Its subcommands provide
all of the core behavior of the toolchain, including compilation, linking, and
developer tools. Each of these has its own subcommand, and you can pass a
specific subcommand to the `help` subcommand to get details about its usage.
)""",
    .help_epilogue = R"""(
For questions, issues, or bug reports, please use our GitHub project:

  https://github.com/carbon-language/carbon-lang
)""",
};

auto Options::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddFlag(
      {
          .name = "verbose",
          .short_name = "v",
          .help = "Enable verbose logging to the stderr stream.",
      },
      [&](CommandLine::FlagBuilder& arg_b) { arg_b.Set(&verbose); });

  b.AddStringOption(
      {
          .name = "runtimes-cache",
          .value_name = "PATH",
          .help = R"""(
Specify a custom runtimes cache location.

By default, the runtimes cache is located in the `carbon_runtimes` subdirectory
of `$XDG_CACHE_HOME` (or `$HOME/.cache` if not set). If unable to use either, it
will be placed in a temporary directory that is removed when the command
completes. This flag overrides that logic with a specific path. It has no effect
if --prebuilt-runtimes is set.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&runtimes_cache_path); });

  b.AddStringOption(
      {
          .name = "prebuilt-runtimes",
          .value_name = "PATH",
          .help = R"""(
Path to prebuilt runtimes tree.

If this option is provided, runtimes will not be built on demand and this path
will be used instead.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&prebuilt_runtimes_path); });
  b.AddFlag(
      {
          .name = "build-runtimes",
          .help = R"""(
Enables on-demand building of target-specific runtimes.

When enabled (the default), any link actions using `clang` will build the
necessary runtimes on-demand. This build will use any customization it can from
the link command line flags to build the runtimes for the correct target and
with any desired features enabled.

Note: this only has an effect when `--prebuilt-runtimes` are not provided. If
there are no prebuilt runtimes and building runtimes is disabled, then it is
assumed the installed toolchain has had the necessary target runtimes added to
the installation tree in the default searched locations.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&build_runtimes_on_demand);
      });

  b.AddFlag(
      {
          .name = "fuzzing",
          .help = "Configure the command line for fuzzing.",
      },
      [&](CommandLine::FlagBuilder& arg_b) { arg_b.Set(&fuzzing); });

  b.AddFlag(
      {
          .name = "include-diagnostic-kind",
          .help = R"""(
When printing diagnostics, include the diagnostic kind as part of output. This
applies to each message that forms a diagnostic, not just the primary message.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&include_diagnostic_kind); });

  b.AddFlag(
      {
          .name = "threads",
          .help = R"""(
Controls whether threads are used to build runtimes.

When enabled (the default), Carbon will try to build runtime libraries using
threads to parallelize the operation. How many threads is controlled
automatically by the system.

Disabling threads ensures a single threaded build of the runtimes which can help
when there are errors or other output.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&threads);
      });

  runtimes.AddTo(b, &selected_subcommand);
  build.AddTo(b, &selected_subcommand);
  clang.AddTo(b, &selected_subcommand);
  compile.AddTo(b, &selected_subcommand);
  config.AddTo(b, &selected_subcommand);
  format.AddTo(b, &selected_subcommand);
  language_server.AddTo(b, &selected_subcommand);
  link.AddTo(b, &selected_subcommand);
  lld.AddTo(b, &selected_subcommand);
  llvm.AddTo(b, &selected_subcommand);

  b.RequiresSubcommand();
}

static auto HandleRuntimesOptions(const Options& options, DriverEnv& driver_env)
    -> bool {
  if (!options.prebuilt_runtimes_path.empty()) {
    auto result = Runtimes::OpenExisting(options.prebuilt_runtimes_path.str(),
                                         driver_env.vlog_stream);
    if (!result.ok()) {
      // TODO: We should provide a better diagnostic than the raw error.
      CARBON_DIAGNOSTIC(DriverPrebuiltRuntimesInvalid, Error, "{0}",
                        std::string);
      driver_env.emitter.Emit(DriverPrebuiltRuntimesInvalid,
                              result.error().message());
      return false;
    }
    driver_env.prebuilt_runtimes = *std::move(result);
    return true;
  }

  if (!options.build_runtimes_on_demand) {
    // Nothing else needed if we're not building runtimes on demand.
    CARBON_CHECK(!driver_env.build_runtimes_on_demand);
    return true;
  }
  driver_env.build_runtimes_on_demand = true;

  // If we don't have prebuilt runtimes and are building them on demand, we need
  // to configure the runtimes cache.
  auto cache_result =
      options.runtimes_cache_path.empty()
          ? Runtimes::Cache::MakeSystem(*driver_env.installation,
                                        driver_env.vlog_stream)
          : Runtimes::Cache::MakeCustom(
                *driver_env.installation,
                std::filesystem::absolute(options.runtimes_cache_path.str()),
                driver_env.vlog_stream);
  if (!cache_result.ok()) {
    // TODO: We should provide a better diagnostic than the raw error.
    CARBON_DIAGNOSTIC(DriverRuntimesCacheInvalid, Error, "{0}", std::string);
    driver_env.emitter.Emit(DriverRuntimesCacheInvalid,
                            cache_result.error().message());
    return false;
  }
  driver_env.runtimes_cache = std::move(*cache_result);
  return true;
}

auto Driver::RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> DriverResult {
  PrettyStackTraceFunction trace_version([&](llvm::raw_ostream& out) {
    out << "Carbon version: " << Version::String << "\n";
  });

  Options options;
  Diagnostics::StreamConsumer consumer(error_stream_);
  DriverEnv env(fs_, installation_, input_stream_, output_stream_,
                error_stream_, fuzzing_, enable_leaking_, &consumer);
  env.mem_usage = mem_usage_;

  ErrorOr<CommandLine::ParseResult> result = CommandLine::Parse(
      args, *env.output_stream, Options::Info,
      [&](CommandLine::CommandBuilder& b) { options.Build(b); });

  // Regardless of whether the parse succeeded, try to use the diagnostic kind
  // flag.
  consumer.set_include_diagnostic_kind(options.include_diagnostic_kind);

  if (env.installation->error()) {
    CARBON_DIAGNOSTIC(DriverInstallInvalid, Error, "{0}", std::string);
    env.emitter.Emit(DriverInstallInvalid, env.installation->error()->str());
    return {.success = false};
  }

  if (!result.ok()) {
    CARBON_DIAGNOSTIC(DriverCommandLineParseFailed, Error, "{0}", std::string);
    env.emitter.Emit(DriverCommandLineParseFailed,
                     PrintToString(result.error()));
    return {.success = false};
  } else if (*result == CommandLine::ParseResult::MetaSuccess) {
    return {.success = true};
  }

  if (!HandleRuntimesOptions(options, env)) {
    return {.success = false};
  }

  if (options.verbose) {
    // Note this implies streamed output in order to interleave.
    env.vlog_stream = env.error_stream;
  }
  if (options.fuzzing) {
    env.fuzzing = true;
  }

  llvm::SingleThreadExecutor single_thread({.ThreadsRequested = 1});
  std::optional<llvm::DefaultThreadPool> threads;
  env.thread_pool = &single_thread;
  if (options.threads) {
    threads.emplace(llvm::optimal_concurrency());
    env.thread_pool = &*threads;
  }

  CARBON_CHECK(options.selected_subcommand != nullptr);
  return options.selected_subcommand->Run(env);
}

}  // namespace Carbon
