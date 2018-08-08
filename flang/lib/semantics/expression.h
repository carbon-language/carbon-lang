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
  using KindParam = std::int64_t;

  ExpressionAnalyzer(evaluate::FoldingContext &c, KindParam dIK)
    : context_{c}, defaultIntegerKind_{dIK} {}

  evaluate::FoldingContext &context() { return context_; }
  KindParam defaultIntegerKind() const { return defaultIntegerKind_; }

  // Performs semantic checking on an expression.  If successful,
  // returns its typed expression representation.
  std::optional<evaluate::GenericExpr> Analyze(const parser::Expr &);
  KindParam Analyze(const std::optional<parser::KindParam> &,
      KindParam defaultKind, KindParam kanjiKind = -1 /* not allowed here */);

private:
  evaluate::FoldingContext context_;
  KindParam defaultIntegerKind_{4};
};
}  // namespace Fortran::semantics
#endif  // FORTRAN_SEMANTICS_EXPRESSION_H_
