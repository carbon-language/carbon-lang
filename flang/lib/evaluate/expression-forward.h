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

#ifndef FORTRAN_EVALUATE_EXPRESSION_FORWARD_H_
#define FORTRAN_EVALUATE_EXPRESSION_FORWARD_H_

// Some forward definitions for expression.h that need to be available
// in variable.h to resolve cases of mutual references between class
// definitions.

#include "type.h"

namespace Fortran::evaluate {

// An expression of some specific result type.
template<typename A> class Expr;
template<int KIND> using IntegerExpr = Expr<Type<Category::Integer, KIND>>;
using DefaultIntegerExpr = IntegerExpr<DefaultInteger::kind>;
template<int KIND> using RealExpr = Expr<Type<Category::Real, KIND>>;
template<int KIND> using ComplexExpr = Expr<Type<Category::Complex, KIND>>;
template<int KIND> using CharacterExpr = Expr<Type<Category::Character, KIND>>;
using LogicalExpr = Expr<Type<Category::Logical, 1>>;

// An expression whose result is of a particular type category and
// any supported kind.
template<Category CAT> struct CategoryExpr;
using GenericIntegerExpr = CategoryExpr<Category::Integer>;
using GenericRealExpr = CategoryExpr<Category::Real>;
using GenericComplexExpr = CategoryExpr<Category::Complex>;
using GenericCharacterExpr = CategoryExpr<Category::Character>;

// A completely generic expression.
struct GenericExpr;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_FORWARD_H_
