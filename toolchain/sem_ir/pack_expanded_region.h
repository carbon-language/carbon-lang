// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_PACK_EXPANDED_REGION_H_
#define CARBON_TOOLCHAIN_SEM_IR_PACK_EXPANDED_REGION_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// The region of code that is expanded by a pack expansion, such as the body of
// a `...` statement. The region is a generic whose final compile-time binding
// is the variadic index; the region is executed once for each value of that
// index, in a specific of the generic.
//
// While this is a nested scope with its own generic, it is not an Entity like
// most declarations, with a name and parameters, so it does not inherit
// EntityWithParamsBase.
struct PackExpandedRegion : Printable<PackExpandedRegion> {
  // The `PackExpansion` instruction that introduces this region.
  InstId decl_id;
  // The generic for the region.
  GenericId generic_id;
  // The variadic index binding. This is the last binding in the generic.
  InstId index_id;
  // The blocks that make up the region. The first block contains the index
  // binding and a `Branch` to the rest of the region, which is the `inst_id`
  // of the `PackExpansion` at `decl_id`.
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

using PackExpandedRegionStore =
    ValueStore<PackExpandedRegionId, PackExpandedRegion, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class ValueStore<SemIR::PackExpandedRegionId,
                                 SemIR::PackExpandedRegion,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_PACK_EXPANDED_REGION_H_
