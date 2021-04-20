// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/paren_contents.h"

namespace Carbon {

const Expression* ParenContents::AsExpression(int line_number) const {
  if (fields_.size() == 1 && fields_.front().name == "" &&
      has_trailing_comma_ == HasTrailingComma::No) {
    return fields_.front().expression;
  } else {
    return AsTuple(line_number);
  }
}

const Expression* ParenContents::AsTuple(int line_number) const {
  auto vec =
      new std::vector<std::pair<std::string, const Carbon::Expression*>>();
  for (const FieldInitializer& initializer : fields_) {
    vec->push_back({initializer.name, initializer.expression});
  }
  return MakeTuple(line_number, vec);
}

}  // namespace Carbon
