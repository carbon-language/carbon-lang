// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_VTABLE_H_
#define CARBON_TOOLCHAIN_SEM_IR_VTABLE_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Vtable-specific fields.
struct VtableFields {
  // The class this is the vtable for.
  ClassId class_id;

  // A block containing all the virtual functions in this class and
  // non-overriden functions in base classes, forming the complete vtable for
  // the class.
  InstBlockId virtual_functions_id;
};

struct Vtable : public VtableFields, public Printable<Vtable> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{class: " << class_id << ", {";
    virtual_functions_id.Print(out);
    out << "}}";
  }
};

using VtableStore = ValueStore<VtableId, Vtable>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_VTABLE_H_
