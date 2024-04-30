// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_CONSTANT_H_
#define CARBON_TOOLCHAIN_LOWER_CONSTANT_H_

#include "llvm/ADT/ArrayRef.h"
#include "toolchain/lower/file_context.h"

namespace Carbon::Lower {

// Forms LLVM constant values for all constants used in the file described by
// `file_context`. The indexes in the `constants` array corresponding to
// template constant instructions are populated with corresponding constant
// values.
auto LowerConstants(FileContext& file_context,
                    llvm::MutableArrayRef<llvm::Constant*> constants) -> void;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_CONSTANT_H_
