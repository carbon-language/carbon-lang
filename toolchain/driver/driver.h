// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_H_

#include "common/command_line.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"

namespace Carbon {

// Command line interface driver.
//
// Provides simple API to parse and run command lines for Carbon.  It is
// generally expected to be used to implement command line tools for working
// with the language.
class Driver {
 public:
  // Constructs a driver with the provided environment. `input_stream` is
  // optional; other parameters are required.
  explicit Driver(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                  const InstallPaths* installation, FILE* input_stream,
                  llvm::raw_pwrite_stream* output_stream,
                  llvm::raw_pwrite_stream* error_stream, bool fuzzing = false,
                  bool enable_leaking = false)
      : fs_(std::move(fs)),
        installation_(installation),
        input_stream_(input_stream),
        output_stream_(output_stream),
        error_stream_(error_stream),
        fuzzing_(fuzzing),
        enable_leaking_(enable_leaking) {}

  // Parses the given arguments into both a subcommand to select the operation
  // to perform and any arguments to that subcommand.
  //
  // Returns true if the operation succeeds. If the operation fails, returns
  // false and any information about the failure is printed to the registered
  // error stream (stderr by default).
  auto RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> DriverResult;

  // Sets a memory-usage sink for subsequent commands. When set, a compile
  // command collects each compiled file's memory usage into `mem_usage` (the
  // same data `--dump-mem-usage` reports), so it can be queried
  // programmatically. The labels of files compiled together are accumulated
  // into the one `mem_usage`; consumers typically sum entries by label. Pass
  // null to disable.
  auto set_mem_usage(MemUsage* mem_usage) -> void { mem_usage_ = mem_usage; }

 private:
  // We store the initial values in the `DriverEnv` that will be used for each
  // subcommand invocation here. These are used as the _starting_ values of the
  // environment, but individual `RunCommand` invocations may customize the
  // `DriverEnv` instance changing these values.
  //
  // For details on each of these fields, see the documentation in `DriverEnv`.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs_;
  const InstallPaths* installation_;
  FILE* input_stream_;
  llvm::raw_pwrite_stream* output_stream_;
  llvm::raw_pwrite_stream* error_stream_;
  bool fuzzing_;
  bool enable_leaking_;
  MemUsage* mem_usage_ = nullptr;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_H_
