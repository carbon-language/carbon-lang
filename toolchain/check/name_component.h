// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_NAME_COMPONENT_H_
#define CARBON_TOOLCHAIN_CHECK_NAME_COMPONENT_H_

#include "toolchain/check/node_stack.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

class Context;

// A component in a declaration name, such as `C[T:! type](N:! T)` in
// `fn C[T:! type](N:! T).F() {}`.
struct NameComponent {
  // The name of the declaration.
  Parse::NodeId name_loc_id;
  SemIR::NameId name_id;

  // The implicit parameter list.
  Parse::NodeId implicit_params_loc_id;
  SemIR::InstBlockId implicit_params_id;

  // The explicit parameter list.
  Parse::NodeId params_loc_id;
  SemIR::InstBlockId params_id;
};

// Pop a name component from the node stack.
auto PopNameComponent(Context& context) -> NameComponent;

// Pop the name of a declaration from the node stack, and diagnose if it has
// parameters.
auto PopNameComponentWithoutParams(Context& context, Lex::TokenKind introducer)
    -> NameComponent;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_NAME_COMPONENT_H_
