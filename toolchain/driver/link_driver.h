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

class LinkDriver {
 public:
  explicit LinkDriver(LinkOptions* options);
  auto Link(DriverEnv& driver_env) -> DriverResult;

 private:
  LinkOptions* options_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_LINK_DRIVER_H_
