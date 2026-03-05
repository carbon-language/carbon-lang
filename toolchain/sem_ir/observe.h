// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_OBSERVE_H_
#define CARBON_TOOLCHAIN_SEM_IR_OBSERVE_H_

#include "toolchain/base/block_value_store.h"
#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// An observe declaration from an interface or function, written
// `observe T == U` or `observe T impls Z`.
//
// While this comes from a `observe` declaration, it is not an Entity like most
// other declarations, with a name and parameters, so it does not inherit
// EntityWithParamsBase.
struct Observe : Printable<Observe> {
  // The location of the `observe` declaration.
  InstId decl_id;
  // The block which contains the `ObserveEquivalent` and `ObserveImpls`
  // operators.
  InstBlockId operations_id;
  // The scope which contains the `observe` declaration.
  NameScopeId parent_scope_id;
  // The function which contains the `observe` declaration.
  InstId parent_function_inst_id;

  auto IsInFunction() const -> bool {
    CARBON_CHECK(parent_scope_id.has_value() !=
                 parent_function_inst_id.has_value());
    return !parent_scope_id.has_value() && parent_function_inst_id.has_value();
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << '{';
    out << "decl_id: " << decl_id << ", operations_id: " << operations_id
        << ", parent_scope: " << parent_scope_id
        << ", parent_function_inst_id: " << parent_function_inst_id;
    out << '}';
  }
};

using ObserveStore = ValueStore<ObserveId, Observe, Tag<CheckIRId>>;
using ObserveBlockStore =
    BlockValueStore<ObserveBlockId, ObserveId, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_OBSERVE_H_
