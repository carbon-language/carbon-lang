// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_PACK_EXPANSION_H_
#define CARBON_TOOLCHAIN_SEM_IR_PACK_EXPANSION_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A pack expansion, such as a `...` statement. The body of the pack expansion
// is a generic whose final compile-time binding is the variadic index; the
// body is executed once for each value of that index, in a specific of the
// generic.
//
// While this is a nested scope with its own generic, it is not an Entity like
// most declarations, with a name and parameters, so it does not inherit
// EntityWithParamsBase.
struct PackExpansion : Printable<PackExpansion> {
  // The `PackExpansionDecl` instruction that introduces this pack expansion.
  InstId decl_id;
  // The generic for the body of the pack expansion.
  GenericId generic_id;
  // The variadic index binding. This is the last binding in the generic.
  InstId index_id;
  // The blocks that make up the body of the pack expansion. The first block
  // contains the index binding and a `Branch` to the rest of the body, which is
  // the `entry_id` of the `PackExpansionDecl` at `decl_id`.
  llvm::SmallVector<InstBlockId> body_block_ids = {};

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{decl_id: " << decl_id << ", generic_id: " << generic_id
        << ", index_id: " << index_id << ", body: [";
    llvm::ListSeparator sep;
    for (auto block_id : body_block_ids) {
      out << sep << block_id;
    }
    out << "]}";
  }
};

using PackExpansionStore =
    ValueStore<PackExpansionId, PackExpansion, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class ValueStore<SemIR::PackExpansionId, SemIR::PackExpansion,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_PACK_EXPANSION_H_
