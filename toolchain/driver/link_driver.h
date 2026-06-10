// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_LINK_DRIVER_H_
#define CARBON_TOOLCHAIN_DRIVER_LINK_DRIVER_H_

#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/link_options.h"

namespace Carbon {

// Helper class to link object files into an output binary using `clang`. Used
// by the `build` and `link` subcommands.
class LinkDriver {
 public:
  explicit LinkDriver(LinkOptions* options);

  // Link the input binaries to the output binary with the configuration
  // specified in the `LinkOptions` provided at construction time.
  auto Link(DriverEnv& driver_env) -> DriverResult;

 private:
  LinkOptions* options_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LINK_DRIVER_H_
