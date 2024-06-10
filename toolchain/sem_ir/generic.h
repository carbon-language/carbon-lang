// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_
#define CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Information for a generic entity, such as a generic class, a generic
// interface, or generic function.
struct Generic : public Printable<Generic> {
  // A portion of the generic corresponding to either the declaration or the
  // definition. These are tracked separately because they're built and resolved
  // at different times.
  struct Fragment {
    // A block containing symbolic constants that are used in this fragment.
    InstBlockId symbolic_constants_id = InstBlockId::Invalid;
    // TODO: Also track:
    // - Types required to be complete in this generic.
    // - Template-dependent instructions in this generic.
  };

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{decl: " << decl_id << ", bindings: " << bindings_id << "}";
  }

  // The following members always have values, and do not change throughout the
  // lifetime of the generic.

  // The first declaration of the generic entity.
  InstId decl_id;
  // A block containing the IDs of compile time bindings in this generic scope.
  // The index in this block will match the `bind_index` of the instruction.
  InstBlockId bindings_id;

  // The portion of the generic corresponding to the declaration of the entity.
  Fragment decl;
  // The portion of the generic corresponding to the definition of the entity.
  Fragment definition;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_GENERIC_H_
