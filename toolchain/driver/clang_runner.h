// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_CLANG_RUNNER_H_
#define CARBON_TOOLCHAIN_DRIVER_CLANG_RUNNER_H_

#include "clang/Basic/DiagnosticIDs.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

// Runs Clang in a similar fashion to invoking it with the provided arguments on
// the command line. We use a textual command line interface to allow easily
// incorporating custom command line flags from user invocations that we don't
// parse, but will pass transparently along to Clang itself.
//
// This doesn't literally use a subprocess to invoke Clang; it instead tries to
// directly use the Clang command line driver library. We also work to simplify
// how that driver operates and invoke it in an opinionated way to get the best
// behavior for our expected use cases in the Carbon driver:
//
// - Minimize canonicalization of file names to try to preserve the paths as
//   users type them.
// - Minimize the use of subprocess invocations which are expensive on some
//   operating systems. To the extent possible, we try to directly invoke the
//   Clang logic within this process.
// - Provide programmatic API to control defaults of Clang. For example, causing
//   verbose output.
//
// Note that this makes the current process behave like running Clang -- it uses
// standard output and standard error, and otherwise can only read and write
// files based on their names described in the arguments. It doesn't provide any
// higher-level abstraction such as streams for inputs or outputs.
class ClangRunner {
 public:
  // Build a Clang runner that uses the provided `exe_name` and `err_stream`.
  //
  // If `verbose` is passed as true, will enable verbose logging to the
  // `err_stream` both from the runner and Clang itself.
  ClangRunner(llvm::StringRef exe_name, llvm::StringRef target,
              llvm::raw_ostream* vlog_stream = nullptr);

  // Run Clang with the provided arguments.
  auto Run(llvm::ArrayRef<llvm::StringRef> args) -> bool;

 private:
  llvm::StringRef exe_name_;
  std::string exe_path_;
  llvm::StringRef target_;
  llvm::raw_ostream* vlog_stream_;

  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagnostic_ids_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_CLANG_RUNNER_H_
