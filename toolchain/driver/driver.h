// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_H_

#include <cstdint>

#include "common/argparse.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Command line interface driver.
//
// Provides simple API to parse and run command lines for Carbon.  It is
// generally expected to be used to implement command line tools for working
// with the language.
class Driver {
 public:
  // Default constructed driver uses stderr for all error and informational
  // output.
  Driver() : output_stream_(llvm::outs()), error_stream_(llvm::errs()) {}

  // Constructs a driver with any error or informational output directed to a
  // specified stream.
  Driver(llvm::raw_ostream& output_stream, llvm::raw_ostream& error_stream)
      : output_stream_(output_stream), error_stream_(error_stream) {}

  // Parses the given arguments into both a subcommand to select the operation
  // to perform and any arguments to that subcommand.
  //
  // Returns true if the operation succeeds. If the operation fails, returns
  // false and any information about the failure is printed to the registered
  // error stream (stderr by default).
  auto RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> bool;

 private:
  enum class Subcommands {
    Compile,
  };

  llvm::raw_ostream& output_stream_;
  llvm::raw_ostream& error_stream_;
  llvm::raw_ostream* vlog_stream_ = nullptr;

  auto RunCompileSubcommand(SubcommandArgs<Subcommands> args) -> bool;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_H_
