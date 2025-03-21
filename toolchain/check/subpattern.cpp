// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/subpattern.h"

#include "toolchain/check/inst.h"

namespace Carbon::Check {

auto BeginSubpattern(Context& context) -> void {
  context.inst_block_stack().Push();
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
}

auto EndSubpatternAsExpr(Context& context, SemIR::InstId result_id)
    -> SemIR::ExprRegionId {
  if (context.region_stack().PeekRegion().size() > 1) {
    // End the exit block with a branch to a successor block, whose contents
    // will be determined later.
    AddInst(context,
            SemIR::LocIdAndInst::NoLoc<SemIR::Branch>(
                {.target_id = context.inst_blocks().AddPlaceholder()}));
  } else {
    // This single-block region will be inserted as a SpliceBlock, so we don't
    // need control flow out of it.
  }
  auto block_id = context.inst_block_stack().Pop();
  CARBON_CHECK(block_id == context.region_stack().PeekRegion().back());

  // TODO: Is it possible to validate that this region is genuinely
  // single-entry, single-exit?
  return context.sem_ir().expr_regions().Add(
      {.block_ids = context.region_stack().PopRegion(),
       .result_id = result_id});
}

auto EndSubpatternAsNonExpr(Context& context) -> void {
  auto block_id = context.inst_block_stack().Pop();
  CARBON_CHECK(block_id == context.region_stack().PeekRegion().back());
  CARBON_CHECK(context.region_stack().PeekRegion().size() == 1);
  // TODO: Add `CARBON_CHECK(inst_blocks().Get(block_id).empty())`.
  // Currently that can fail when ending a tuple pattern in a name binding
  // decl in a class or interface.
  context.region_stack().PopAndDiscardRegion();
}

}  // namespace Carbon::Check
