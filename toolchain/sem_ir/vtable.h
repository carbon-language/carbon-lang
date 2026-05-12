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

  // Specifies that this vtable uses Carbon's native vtable layout, rather than
  // an Itanium-compatible vtable layout for C++ interop.
  // TODO: This might change to an enum representing more diverse vtable layouts
  // (eg: Carbon type with a C++ vtable distinct from a C++ type with a C++
  // vtable, maybe other language interop, etc)
  bool carbon_native_vtable = true;
};

struct Vtable : public VtableFields, public Printable<Vtable> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{class: " << class_id << ", {";
    virtual_functions_id.Print(out);
    out << "}}";
  }
};

using VtableStore = ValueStore<VtableId, Vtable, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_VTABLE_H_
