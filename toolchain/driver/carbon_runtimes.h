// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_CARBON_RUNTIMES_H_
#define CARBON_TOOLCHAIN_DRIVER_CARBON_RUNTIMES_H_

#include <filesystem>
#include <memory>

#include "common/error.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/codegen_options.h"
#include "toolchain/driver/compile_driver.h"
#include "toolchain/driver/compile_options.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/runtimes_cache.h"

namespace Carbon {

struct CompileOptions;

// Common code for Carbon runtimes builders.
//
// Although this class has only a single derivation for building the prelude,
// we anticipate wanting to build other parts of Core in the future and so
// design for extensibility.
// TODO: Build all parts of the Core, not just the prelude.
class CarbonRuntimesBuilderBase {
 protected:
  CarbonRuntimesBuilderBase(DriverEnv* driver_env,
                            const CodegenOptions* codegen_options);

  // We use protected members as this base is just factoring out common
  // implementation details of other runners.
  //
  // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
  //
  CompileOptions compile_options_;
  CompileDriver compile_driver_;
  DriverEnv* driver_env_;
  // Base path for the input source files. The enumerated constant list of
  // paths is all relative to this base path.
  std::filesystem::path install_root_;
  // Output subpath for the built binaries.
  std::filesystem::path lib_path_;
  ErrorOr<std::filesystem::path> result_;
  std::optional<Runtimes::Builder> runtimes_builder_;
  // NOLINTEND(misc-non-private-member-variables-in-classes)
};

class CarbonPreludeBuilder : public CarbonRuntimesBuilderBase {
 public:
  CarbonPreludeBuilder(DriverEnv* driver_env, Runtimes* runtimes,
                       const CodegenOptions* codegen_options);
  auto Build() && -> ErrorOr<std::filesystem::path>;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_CARBON_RUNTIMES_H_
