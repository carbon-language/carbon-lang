// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_DEFAULT_VALUE_H_
#define CARBON_TOOLCHAIN_SEM_IR_DEFAULT_VALUE_H_

#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Information about a default value for a pattern, such as default values for
// function parameters.
struct DefaultValue : public Printable<DefaultValue> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{raw_id: " << raw_id << ", value_id: " << value_id << "}";
  }

  // The instruction specifying the default value as specified by the developer,
  // before conversion to the pattern scrutinee type is applied.
  InstId raw_id;

  // The instruction specifying the default value in the same type as the
  // scrutinee type.
  InstId value_id;

  // True if the user left this default value unspecified. We still store these
  // so the location of the unspecified default value can be used in
  // diagnostics.
  bool is_unspecified;
};

class DefaultValueStore
    : public ValueStore<DefaultValueId, DefaultValue, Tag<SemIR::CheckIRId>> {
 public:
  using ValueStore::ValueStore;
};

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class ValueStore<SemIR::DefaultValueId, SemIR::DefaultValue,
                                 Tag<SemIR::CheckIRId>>;
}

#endif  // CARBON_TOOLCHAIN_SEM_IR_DEFAULT_VALUE_H_