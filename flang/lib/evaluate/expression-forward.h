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
template<int KIND> using IntegerExpr = Expr<Type<TypeCategory::Integer, KIND>>;
using DefaultIntegerExpr = IntegerExpr<DefaultInteger::kind>;
template<int KIND> using RealExpr = Expr<Type<TypeCategory::Real, KIND>>;
template<int KIND> using ComplexExpr = Expr<Type<TypeCategory::Complex, KIND>>;
template<int KIND>
using CharacterExpr = Expr<Type<TypeCategory::Character, KIND>>;
template<int KIND> using LogicalExpr = Expr<Type<TypeCategory::Logical, KIND>>;

// An expression whose result is within one particular type category and
// of any supported kind.
using AnyKindIntegerExpr = Expr<AnyKindType<TypeCategory::Integer>>;
using AnyKindRealExpr = Expr<AnyKindType<TypeCategory::Real>>;
using AnyKindComplexExpr = Expr<AnyKindType<TypeCategory::Complex>>;
using AnyKindCharacterExpr = Expr<AnyKindType<TypeCategory::Character>>;
using AnyKindLogicalExpr = Expr<AnyKindType<TypeCategory::Logical>>;

// A completely generic expression.
struct GenericExpr;

}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_EXPRESSION_FORWARD_H_
