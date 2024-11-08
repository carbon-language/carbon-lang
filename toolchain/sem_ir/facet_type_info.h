// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
#define CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct FacetTypeInfo : Printable<FacetTypeInfo> {
  // TODO: Need to switch to a processed, canonical form, that can support facet
  // type equality as defined by
  // https://github.com/carbon-language/carbon-lang/issues/2409.
  TypeId base_facet_type_id;
  InstBlockId requirement_block_id;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{base_facet_type: " << base_facet_type_id
        << ", requirements: " << requirement_block_id << "}";
  }
  auto operator==(const FacetTypeInfo& rhs) const -> bool {
    return base_facet_type_id == rhs.base_facet_type_id &&
           requirement_block_id == rhs.requirement_block_id;
  }
  // TODO: Implement hash
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FACET_TYPE_INFO_H_
