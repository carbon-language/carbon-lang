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

struct DriverEnv {
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
  bool fuzzing = false;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_ENV_H_
