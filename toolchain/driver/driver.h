// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_DRIVER_H_
#define CARBON_TOOLCHAIN_DRIVER_DRIVER_H_

#include <cstdint>

#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon {

// Command line interface driver.
//
// Provides simple API to parse and run command lines for Carbon.  It is
// generally expected to be used to implement command line tools for working
// with the language.
class Driver {
 public:
  // Constructs a driver with any error or informational output directed to a
  // specified stream.
  Driver(llvm::vfs::FileSystem& fs, llvm::raw_ostream& output_stream,
         llvm::raw_ostream& error_stream)
      : fs_(fs), output_stream_(output_stream), error_stream_(error_stream) {
    (void)fs_;
  }

  // Parses the given arguments into both a subcommand to select the operation
  // to perform and any arguments to that subcommand.
  //
  // Returns true if the operation succeeds. If the operation fails, returns
  // false and any information about the failure is printed to the registered
  // error stream (stderr by default).
  auto RunFullCommand(llvm::ArrayRef<llvm::StringRef> args) -> bool;

  // Subcommand that prints available help text to the error stream.
  //
  // Optionally one positional parameter may be provided to select a particular
  // subcommand or detailed section of help to print.
  //
  // Returns true if appropriate help text was found and printed. If an invalid
  // positional parameter (or flag) is provided, returns false.
  auto RunHelpSubcommand(DiagnosticConsumer& consumer,
                         llvm::ArrayRef<llvm::StringRef> args) -> bool;

  // Subcommand that dumps internal compilation information for the provided
  // source file.
  //
  // Requires exactly one positional parameter to designate the source file to
  // read. May be `-` to read from stdin.
  //
  // Returns true if the operation succeeds. If the operation fails, this
  // returns false and any information about the failure is printed to the
  // registered error stream (stderr by default).
  auto RunDumpSubcommand(DiagnosticConsumer& consumer,
                         llvm::ArrayRef<llvm::StringRef> args) -> bool;

 private:
  auto ReportExtraArgs(llvm::StringRef subcommand_text,
                       llvm::ArrayRef<llvm::StringRef> args) -> void;

  llvm::vfs::FileSystem& fs_;
  llvm::raw_ostream& output_stream_;
  llvm::raw_ostream& error_stream_;
  llvm::raw_ostream* vlog_stream_ = nullptr;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_DRIVER_H_
