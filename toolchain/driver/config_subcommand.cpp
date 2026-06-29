// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/config_subcommand.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearch.h"
#include "common/check.h"
#include "common/command_line.h"
#include "common/filesystem.h"
#include "common/version.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/base/clang_invocation.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"

namespace Carbon {

auto ConfigOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddFlag(
      {
          .name = "json",
          .help = R"""(
Render output as a JSON map for easy parsing.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(false);
        arg_b.Set(&json_output);
      });

  codegen_options.Build(b);
}

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "config",
    .help = R"""(
Print configuration info for the Carbon toolchain.

This subcommand displays configuration information for the Carbon toolchain.
This can be useful for build systems as well as debugging issues.
)""",
};

ConfigSubcommand::ConfigSubcommand() : DriverSubcommand(SubcommandInfo) {}

namespace {
struct ConfigDataEntry {
  std::string key;
  std::variant<std::string, llvm::SmallVector<std::string>> value;
};
}  // namespace

// Creates a Clang invocation and queries it for config data to render.
//
// This includes the sysroot and the include directories searched during
// compilation.
//
// If there are any errors setting up Clang, this will diagnose them using
// `driver_env.consumer` and return `false`. If successful, returns `true`.
static auto ComputeClangConfig(DriverEnv& driver_env,
                               llvm::StringRef target_str,
                               llvm::SmallVectorImpl<ConfigDataEntry>& data)
    -> bool {
  // Build a library invocation of Clang in order to query its header search
  // paths.
  std::shared_ptr clang_invocation =
      BuildClangInvocation(*driver_env.consumer, driver_env.fs,
                           *driver_env.installation, target_str, {});
  clang_invocation->getFrontendOpts().DisableFree = false;

  // Setup up a driver-style diagnostic engine for the compiler invocation and
  // instance below as we won't go past that while computing the include dirs.
  Diagnostics::ErrorTrackingConsumer error_tracker(*driver_env.consumer);
  Diagnostics::NoLocEmitter emitter(&error_tracker);
  ClangDriverDiagnosticConsumer diagnostic_consumer(&emitter);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(
      clang::CompilerInstance::createDiagnostics(
          *driver_env.fs, clang_invocation->getDiagnosticOpts(),
          &diagnostic_consumer,
          /*ShouldOwnClient=*/false));

  auto clang_instance =
      std::make_unique<clang::CompilerInstance>(clang_invocation);
  clang_instance->setDiagnostics(diags);
  clang_instance->setVirtualFileSystem(driver_env.fs);
  clang_instance->createFileManager();
  clang_instance->createSourceManager();
  if (!clang_instance->createTarget()) {
    CARBON_DIAGNOSTIC(ConfigFailedToSetupTarget, Error,
                      "unable to setup the requested target `{0}`",
                      std::string);
    driver_env.emitter.Emit(ConfigFailedToSetupTarget, target_str.str());
    return false;
  }

  auto header_search = std::make_unique<clang::HeaderSearch>(
      clang_instance->getHeaderSearchOpts(), clang_instance->getSourceManager(),
      clang_instance->getDiagnostics(), clang_instance->getLangOpts(),
      &clang_instance->getTarget());
  clang::ApplyHeaderSearchOptions(
      *header_search, clang_instance->getHeaderSearchOpts(),
      clang_instance->getLangOpts(), clang_instance->getTarget().getTriple());

  // If we ended up diagnosing any errors, just return. They will have been
  // converted to Carbon diagnostics.
  if (error_tracker.seen_error()) {
    return false;
  }

  data.push_back({.key = "CLANG_SYSROOT",
                  .value = clang_instance->getHeaderSearchOpts().Sysroot});

  llvm::SmallVector<std::string> search_paths;
  for (const auto& search_dir : header_search->search_dir_range()) {
    search_paths.push_back(search_dir.getName().str());
  }
  data.push_back(
      {.key = "CLANG_INCLUDE_DIRS", .value = std::move(search_paths)});

  return true;
}

static auto RenderDataAsJson(llvm::ArrayRef<ConfigDataEntry> data,
                             llvm::raw_ostream& out) -> void {
  out << "{\n";
  llvm::ListSeparator data_sep(",\n");
  for (const auto& entry : data) {
    out << data_sep << "    \"" << entry.key << "\": ";

    if (const auto* value = std::get_if<std::string>(&entry.value)) {
      out << "\"" << *value << "\"";
    } else if (const auto* value =
                   std::get_if<llvm::SmallVector<std::string>>(&entry.value)) {
      out << "[\n";
      llvm::ListSeparator element_sep(",\n");
      for (const std::string& value_element : *value) {
        out << element_sep << "        \"" << value_element << "\"";
      }
      out << "\n    ]";
    } else {
      CARBON_FATAL("Invalid value in config data entry!");
    }
  }
  out << "\n}\n";
}

static auto RenderData(llvm::ArrayRef<ConfigDataEntry> data,
                       llvm::raw_ostream& out) -> void {
  for (const auto& entry : data) {
    out << entry.key << ":";
    if (const auto* value = std::get_if<std::string>(&entry.value)) {
      out << " " << *value << "\n";
    } else if (const auto* value =
                   std::get_if<llvm::SmallVector<std::string>>(&entry.value)) {
      out << "\n";
      for (const std::string& value_element : *value) {
        out << "    " << value_element << "\n";
      }
    } else {
      CARBON_FATAL("Invalid value in config data entry!");
    }
  }
}

auto ConfigSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  bool result = true;

  // Start with basic data available from the driver or global constants.
  llvm::SmallVector<ConfigDataEntry> data = {
      {.key = "CLANG_RESOURCE_DIR",
       .value = driver_env.installation->clang_resource_path()},
      {.key = "INSTALL_ROOT", .value = driver_env.installation->root()},
      {.key = "LLVM_BINDIR",
       .value = driver_env.installation->llvm_install_bin()},
      {.key = "VERSION", .value = Version::String.str()},
  };

  // Try to read the installation digest and include that.
  auto read_result = Filesystem::Cwd().ReadFileToString(
      driver_env.installation->digest_path());
  if (!read_result.ok()) {
    CARBON_DIAGNOSTIC(ConfigFailedToReadDigest, Error,
                      "unable to read the installation's digest file: {0}",
                      std::string);
    driver_env.emitter.Emit(ConfigFailedToReadDigest,
                            read_result.error().ToString());

    // Remember that we encountered an error but continue to give a minimally
    // useful `config` output.
    result = false;
  } else {
    data.push_back({.key = "INSTALL_DIGEST",
                    .value = llvm::StringRef(*read_result).rtrim().str()});
  }

  // Compute and print Clang's config entries if we can. This will have been
  // diagnosed while computing, so just track if we hit errors.
  result &=
      ComputeClangConfig(driver_env, options_.codegen_options.target, data);

  llvm::sort(data, [](const ConfigDataEntry& lhs, const ConfigDataEntry& rhs) {
    return lhs.key < rhs.key;
  });

  if (options_.json_output) {
    RenderDataAsJson(data, *driver_env.output_stream);
  } else {
    RenderData(data, *driver_env.output_stream);
  }

  return {.success = result};
}

}  // namespace Carbon
