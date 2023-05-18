// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"
#include "common/argparse.h"

namespace Carbon {

auto ProgramArgs::TestFlag(llvm::StringRef name) const -> bool {
  auto flag_iterator = flags_.find(name);
  CARBON_CHECK(flag_iterator != flags_.end()) << "Flag name not registered: " << name;
  auto [flag_kind, value_index] = flag_iterator->second;
  CARBON_CHECK(flag_kind == FlagKind::Boolean)
      << "Flag named '" << name << "' not a boolean flag";
  return boolean_flag_values_[value_index];
}

}  // namespace Carbon
