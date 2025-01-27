// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_ENV_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_ENV_H_

#include <cstdio>

#include "common/ostream.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// Driver environment information, encapsulated for easy passing to subcommands.
struct DriverEnv {
  explicit DriverEnv(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                     const InstallPaths* installation, FILE* input_stream,
                     llvm::raw_pwrite_stream* output_stream,
                     llvm::raw_pwrite_stream* error_stream, bool fuzzing)
      : fs(std::move(fs)),
        installation(installation),
        input_stream(input_stream),
        output_stream(output_stream),
        error_stream(error_stream),
        fuzzing(fuzzing) {}

  // The filesystem for source code.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs;

  // Helper to locate the toolchain installation's files.
  const InstallPaths* installation;

  // Standard input; stdin. May be null, to prevent accidental use.
  FILE* input_stream;
  // Standard output; stdout.
  llvm::raw_pwrite_stream* output_stream;
  // Error output; stderr.
  llvm::raw_pwrite_stream* error_stream;

  // For CARBON_VLOG.
  llvm::raw_pwrite_stream* vlog_stream = nullptr;

  // Tracks when the driver is being fuzzed. This allows specific commands to
  // error rather than perform operations that aren't well behaved during
  // fuzzing.
  bool fuzzing;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_ENV_H_
