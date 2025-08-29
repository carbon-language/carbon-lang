// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPLICIT_TYPE_IMPLS_H_
#define CARBON_TOOLCHAIN_CHECK_IMPLICIT_TYPE_IMPLS_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Constructs `impl <class> as Destroy { ... }`, with appropriate implementation
// based on the `destroy` function and members.
auto MakeClassDestroyImpl(Context& context, SemIR::ClassId class_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPLICIT_TYPE_IMPLS_H_
