// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/lld_runner.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>

#include "common/vlog.h"
#include "lld/Common/Driver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

// Declare the supported driver flavor entry points.
// NOLINTNEXTLINE(readability-identifier-naming): External library name.
LLD_HAS_DRIVER(elf)
// NOLINTNEXTLINE(readability-identifier-naming): External library name.
LLD_HAS_DRIVER(macho)

namespace Carbon {

LLDRunner::LLDRunner(const InstallPaths* install_paths,
                     llvm::raw_ostream* vlog_stream)
    : ToolRunnerBase(install_paths, vlog_stream) {}

auto LLDRunner::GnuLink(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  // Allocate one chunk of storage for the actual C-strings and a vector of
  // pointers into the storage.
  llvm::OwningArrayRef<char> cstr_arg_storage;
  llvm::SmallVector<const char*, 64> cstr_args = BuildCStrArgs(
      "LLD", installation_->ld_lld_path(), args, cstr_arg_storage, "-v");

  CARBON_VLOG("Running LLD GNU-platform link...\n");
  lld::Result result =
      lld::lldMain(cstr_args, llvm::outs(), llvm::errs(),
                   {lld::DriverDef{.f = lld::Gnu, .d = &lld::elf::link}});

  // Check for an unrecoverable error.
  CARBON_CHECK(result.canRunAgain, "LLD encountered an unrecoverable error!");

  // TODO: Should this be forwarding the full exit code?
  return result.retCode == 0;
}

auto LLDRunner::DarwinLink(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  // Allocate one chunk of storage for the actual C-strings and a vector of
  // pointers into the storage.
  llvm::OwningArrayRef<char> cstr_arg_storage;
  llvm::SmallVector<const char*, 64> cstr_args = BuildCStrArgs(
      "LLD", installation_->ld64_lld_path(), args, cstr_arg_storage, "-v");

  CARBON_VLOG("Running LLD Darwin-platform link...\n");
  lld::Result result =
      lld::lldMain(cstr_args, llvm::outs(), llvm::errs(),
                   {lld::DriverDef{.f = lld::Darwin, .d = &lld::macho::link}});

  // Check for an unrecoverable error.
  CARBON_CHECK(result.canRunAgain, "LLD encountered an unrecoverable error!");

  // TODO: Should this be forwarding the full exit code?
  return result.retCode == 0;
}

}  // namespace Carbon
