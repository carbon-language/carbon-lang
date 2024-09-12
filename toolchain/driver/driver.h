// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_H_

#include "common/command_line.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/driver_env.h"

namespace Carbon {

// Command line interface driver.
//
// Provides simple API to parse and run command lines for Carbon.  It is
// generally expected to be used to implement command line tools for working
// with the language.
class Driver {
 public:
  // The result of RunCommand().
  struct RunResult {
    // Overall success result.
    bool success;

    // Per-file success results. May be empty if files aren't individually
    // processed.
    llvm::SmallVector<std::pair<std::string, bool>> per_file_success;
  };

  // Constructs a driver with any error or informational output directed to a
  // specified stream.
  Driver(llvm::vfs::FileSystem& fs, const InstallPaths* installation,
         llvm::raw_pwrite_stream& output_stream,
         llvm::raw_pwrite_stream& error_stream)
      : driver_env_{.fs = fs,
                    .installation = installation,
                    .output_stream = output_stream,
                    .error_stream = error_stream} {}

  // Parses the given arguments into both a subcommand to select the operation
  // to perform and any arguments to that subcommand.
  //
  // Returns true if the operation succeeds. If the operation fails, returns
  // false and any information about the failure is printed to the registered
  // error stream (stderr by default).
  auto RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> RunResult;

 private:
  struct Options;
  struct CodegenOptions;
  struct CompileOptions;
  struct LinkOptions;
  class CompilationUnit;

  // Delegates to the command line library to parse the arguments and store the
  // results in a custom `Options` structure that the rest of the driver uses.
  auto ParseArgs(llvm::ArrayRef<llvm::StringRef> args, Options& options)
      -> CommandLine::ParseResult;

  // Does custom validation of the compile-subcommand options structure beyond
  // what the command line parsing library supports.
  auto ValidateCompileOptions(const CompileOptions& options) const -> bool;

  // Implements the compile subcommand of the driver.
  auto Compile(const CompileOptions& options,
               const CodegenOptions& codegen_options) -> RunResult;

  // Implements the link subcommand of the driver.
  auto Link(const LinkOptions& options, const CodegenOptions& codegen_options)
      -> RunResult;

  DriverEnv driver_env_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_H_
