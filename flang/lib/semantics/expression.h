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
#include "../evaluate/type.h"
#include <optional>

namespace Fortran::parser {
class Expr;
class Program;
}

namespace Fortran::semantics {

class SemanticsContext;

using MaybeExpr = std::optional<evaluate::Expr<evaluate::SomeType>>;

// Semantic analysis of one expression.
std::optional<evaluate::Expr<evaluate::SomeType>> AnalyzeExpr(
    SemanticsContext &, const parser::Expr &);

// Semantic analysis of all expressions in a parse tree, which is
// decorated with typed representations for top-level expressions.
void AnalyzeExpressions(parser::Program &, SemanticsContext &);
}
#endif  // FORTRAN_SEMANTICS_EXPRESSION_H_
