// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/struct.h"

#include "toolchain/check/context.h"
#include "toolchain/diagnostics/format_providers.h"

namespace Carbon::Check {
auto PopStructFieldNameNodes(Context& context, size_t field_count)
    -> llvm::SmallVector<Parse::NodeId> {
  llvm::SmallVector<Parse::NodeId> nodes;
  nodes.reserve(field_count);
  for ([[maybe_unused]] auto i : llvm::seq(field_count)) {
    auto [name_node, _] =
        context.node_stack().PopWithNodeId<Parse::NodeCategory::MemberName>();
    nodes.push_back(name_node);
  }
  return nodes;
}

// Diagnoses and returns true if there's a duplicate name.
auto DiagnoseDuplicateNames(Context& context,
                            llvm::ArrayRef<Parse::NodeId> field_name_nodes,
                            llvm::ArrayRef<SemIR::StructTypeField> fields,
                            StructKind struct_kind_for_diagnostic) -> bool {
  Map<SemIR::NameId, Parse::NodeId> names;
  for (auto [field_name_node, field] :
       llvm::zip_equal(field_name_nodes, fields)) {
    auto result = names.Insert(field.name_id, field_name_node);
    if (!result.is_inserted()) {
      CARBON_DIAGNOSTIC(
          StructNameDuplicate, Error,
          "duplicated field name `{1}` in "
          "{0:=0:struct literal|=1:struct type literal|=2:struct pattern}",
          Diagnostics::IntAsSelect, SemIR::NameId);
      CARBON_DIAGNOSTIC(StructNamePrevious, Note,
                        "field with the same name here");
      context.emitter()
          .Build(result.value(), StructNameDuplicate,
                 static_cast<int>(struct_kind_for_diagnostic), field.name_id)
          .Note(field_name_node, StructNamePrevious)
          .Emit();
      return true;
    }
  }
  return false;
}

}  // namespace Carbon::Check
