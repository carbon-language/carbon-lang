// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_SEMANTICS_EXPRESSION_H_
#define FORTRAN_SEMANTICS_EXPRESSION_H_

#include "../evaluate/expression.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"
#include <cinttypes>
#include <optional>

namespace Fortran::semantics {

class ExpressionAnalyzer {
public:
  ExpressionAnalyzer(parser::Messages &m, std::uint64_t dIK)
    : messages_{m}, defaultIntegerKind_{dIK} {}
  std::optional<evaluate::GenericExpr> Analyze(const parser::Expr &);
  std::optional<evaluate::GenericExpr> Analyze(
      const parser::IntLiteralConstant &);
  std::optional<evaluate::GenericExpr> Analyze(const parser::LiteralConstant &);

private:
  parser::Messages &messages_;
  const parser::CharBlock at_;
  std::uint64_t defaultIntegerKind_{4};
};
}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_EXPRESSION_H_
