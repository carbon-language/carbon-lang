// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_MAIN_H_
#define CARBON_EXPLORER_MAIN_H_

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace Carbon {

// Runs explorer. relative_prelude_path must be POSIX-style, not native, and
// will be translated to native.
auto ExplorerMain(int argc, char** argv, void* static_for_main_addr,
                  llvm::StringRef relative_prelude_path) -> int;

// Wrapped by the above, but without some process init. This is used directly by
// tests, whereas the above is used directly by main binaries. The
// out_stream_for_trace is only used when `--trace_file=-` is specified.
//
// TODO: We use argc/argv for parameters because command line parsing requires
// it. When that's replaced, we should switch to
// llvm::SmallVector<llvm::StringRef> args because it'll work better with tests.
auto ExplorerMain(int argc, const char** argv, llvm::StringRef install_path,
                  llvm::StringRef relative_prelude_path,
                  llvm::raw_ostream& out_stream, llvm::raw_ostream& err_stream,
                  llvm::raw_ostream& out_stream_for_trace,
                  llvm::vfs::FileSystem& fs) -> int;

}  // namespace Carbon

#endif  // CARBON_EXPLORER_MAIN_H_
