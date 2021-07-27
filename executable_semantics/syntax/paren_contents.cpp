// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/syntax/paren_contents.h"

namespace Carbon {

const Expression* ParenContents::AsExpression(int line_number) const {
  if (fields_.size() == 1 and fields_.front().name == "" and
      has_trailing_comma_ == HasTrailingComma::No) {
    return new Expression(*fields_.front().expression);
  } else {
    return AsTuple(line_number);
  }
}

const Expression* ParenContents::AsTuple(int line_number) const {
  return Expression::MakeTupleLiteral(line_number, fields_);
}

}  // namespace Carbon
