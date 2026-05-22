// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/cpp_initializer_list.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst_categories.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

auto GetStdInitializerListLayout(const File& sem_ir, TypeId type_id)
    -> StdInitializerListLayout {
  auto repr_id = sem_ir.types().GetObjectRepr(type_id);
  if (!repr_id.has_value()) {
    // An incomplete type doesn't have a recognized layout.
    return StdInitializerListLayout{};
  }

  auto struct_type = sem_ir.types().TryGetAs<AnyStructType>(repr_id);
  if (!struct_type) {
    return StdInitializerListLayout{};
  }

  auto fields = sem_ir.struct_type_fields().Get(struct_type->fields_id);
  if (fields.size() != 2) {
    return StdInitializerListLayout{};
  }

  // The first field must be a pointer.
  auto field0_type_id = sem_ir.types().GetTransitiveUnqualifiedAdaptedType(
      sem_ir.types().GetTypeIdForTypeInstId(fields[0].type_inst_id));
  if (!sem_ir.types().Is<PointerType>(field0_type_id.first)) {
    return StdInitializerListLayout{};
  }

  // The second field can be either a pointer or an integer.
  auto field1_type_id = sem_ir.types().GetTransitiveUnqualifiedAdaptedType(
      sem_ir.types().GetTypeIdForTypeInstId(fields[1].type_inst_id));
  if (sem_ir.types().Is<PointerType>(field1_type_id.first)) {
    return StdInitializerListLayout{
        .kind = StdInitializerListLayout::PointerPointer};
  }
  if (sem_ir.types().TryGetIntTypeInfo(field1_type_id.first)) {
    return StdInitializerListLayout{
        .kind = StdInitializerListLayout::PointerInt,
        .size_type_id = field1_type_id.first};
  }

  return StdInitializerListLayout{};
}

}  // namespace Carbon::SemIR
