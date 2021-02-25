// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/ast/expression_or_field_list.h"

namespace Carbon {

auto MakeExp(Expression* exp) -> ExpOrFieldList* {
  auto e = new ExpOrFieldList();
  e->tag = ExpOrFieldListKind::Exp;
  e->u.exp = exp;
  return e;
}

auto MakeFieldList(std::list<std::pair<std::string, Expression*>>* fields)
    -> ExpOrFieldList* {
  auto e = new ExpOrFieldList();
  e->tag = ExpOrFieldListKind::FieldList;
  e->u.fields = fields;
  return e;
}

auto MakeConsField(ExpOrFieldList* e1, ExpOrFieldList* e2) -> ExpOrFieldList* {
  auto fields = new std::list<std::pair<std::string, Expression*>>();
  switch (e1->tag) {
    case ExpOrFieldListKind::Exp:
      fields->push_back(std::make_pair("", e1->u.exp));
      break;
    case ExpOrFieldListKind::FieldList:
      for (auto& field : *e1->u.fields) {
        fields->push_back(field);
      }
      break;
  }
  switch (e2->tag) {
    case ExpOrFieldListKind::Exp:
      fields->push_back(std::make_pair("", e2->u.exp));
      break;
    case ExpOrFieldListKind::FieldList:
      for (auto& field : *e2->u.fields) {
        fields->push_back(field);
      }
      break;
  }
  return MakeFieldList(fields);
}

}  // namespace Carbon
