// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/impl_binding.h"

#include "explorer/ast/pattern.h"

namespace Carbon {

ImplBinding::ImplBinding(CloneContext& context, const ImplBinding& other)
    : AstNode(context, other),
      type_var_(context.Remap(other.type_var_)),
      iface_(context.Clone(other.iface_)),
      symbolic_identity_(context.Clone(other.symbolic_identity_)),
      original_(context.Remap(other.original_)) {}

}  // namespace Carbon
