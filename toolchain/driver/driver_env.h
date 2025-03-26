// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_ENV_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_ENV_H_

#include <cstdio>
#include <utility>

#include "common/ostream.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// Driver environment information, encapsulated for easy passing to subcommands.
struct DriverEnv {
  explicit DriverEnv(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                     const InstallPaths* installation, FILE* input_stream,
                     llvm::raw_pwrite_stream* output_stream,
                     llvm::raw_pwrite_stream* error_stream, bool fuzzing,
                     bool enable_leaking)
      : fs(std::move(fs)),
        installation(installation),
        input_stream(input_stream),
        output_stream(output_stream),
        error_stream(error_stream),
        fuzzing(fuzzing),
        enable_leaking(enable_leaking),
        consumer(error_stream),
        emitter(&consumer) {}

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

  // Tracks when the driver is being fuzzed. This allows specific commands to
  // error rather than perform operations that aren't well behaved during
  // fuzzing.
  bool fuzzing;

  // Tracks whether the driver can leak resources, typically because it is being
  // invoked as part of a single command line program execution. Defaults to
  // `false` for safe and correct library execution.
  bool enable_leaking = false;

  // A diagnostic consumer, to be able to connect output.
  Diagnostics::StreamConsumer consumer;

  // A diagnostic emitter that has no locations.
  Diagnostics::NoLocEmitter emitter;

  // For CARBON_VLOG.
  llvm::raw_pwrite_stream* vlog_stream = nullptr;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_ENV_H_
