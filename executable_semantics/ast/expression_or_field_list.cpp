// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression_or_field_list.h"

namespace Carbon {

auto MakeFieldList(std::list<std::pair<std::string, Expression*>>* fields)
    -> FieldList* {
  auto e = new FieldList();
  e->fields = fields;
  return e;
}

auto MakeConsField(FieldList* e1, FieldList* e2) -> FieldList* {
  auto fields = new std::list<std::pair<std::string, Expression*>>();
  for (auto& field : *e1->fields) {
    fields->push_back(field);
  }
  for (auto& field : *e2->fields) {
    fields->push_back(field);
  }
  auto result = MakeFieldList(fields);
  result->has_explicit_comma = true;
  return result;
}

}  // namespace Carbon
