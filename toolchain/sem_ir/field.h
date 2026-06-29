// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FIELD_H_
#define CARBON_TOOLCHAIN_SEM_IR_FIELD_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A class field.
struct Field : public Printable<Field> {
  ElementIndex index;
  SemIR::InstId initializer_id;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{index: " << index << ", initializer_id: " << initializer_id << "}";
  }
};

using FieldStore = ValueStore<FieldId, Field, Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class ValueStore<SemIR::FieldId, SemIR::Field,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_FIELD_H_
