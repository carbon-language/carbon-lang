// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/lld_runner.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>

#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

// Declare the supported driver flavor entry points.
//
// TODO: Currently, just ELF and MachO, but eventually we should support all of
// the LLD platforms.
//
// NOLINTBEGIN(readability-identifier-naming): External library name.
LLD_HAS_DRIVER(elf)
LLD_HAS_DRIVER(macho)
// NOLINTEND(readability-identifier-naming)

namespace Carbon {

auto LldRunner::LinkHelper(llvm::StringLiteral label,
                           llvm::ArrayRef<llvm::StringRef> args,
                           const std::string& path, lld::DriverDef driver_def)
    -> bool {
  // Allocate one chunk of storage for the actual C-strings and a vector of
  // pointers into the storage.
  llvm::OwningArrayRef<char> cstr_arg_storage;
  llvm::SmallVector<const char*, 64> cstr_args =
      BuildCStrArgs("LLD", path, "-v", args, cstr_arg_storage);

  CARBON_VLOG("Running LLD {0}-platform link...\n", label);
  lld::Result result =
      lld::lldMain(cstr_args, llvm::outs(), llvm::errs(), {driver_def});

  // Check for an unrecoverable error.
  CARBON_CHECK(result.canRunAgain, "LLD encountered an unrecoverable error!");

  // TODO: Should this be forwarding the full exit code?
  return result.retCode == 0;
}

auto LldRunner::ElfLink(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  return LinkHelper("GNU", args, installation_->ld_lld_path(),
                    {.f = lld::Gnu, .d = &lld::elf::link});
}

auto LldRunner::MachOLink(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  return LinkHelper("Darwin", args, installation_->ld64_lld_path(),
                    {.f = lld::Darwin, .d = &lld::macho::link});
}

}  // namespace Carbon
