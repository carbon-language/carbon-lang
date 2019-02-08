// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "constant.h"
#include "expression.h"
#include "type.h"
#include "../parser/characters.h"

namespace Fortran::evaluate {

template<typename RESULT, typename VALUE>
ConstantBase<RESULT, VALUE>::~ConstantBase() {}

template<typename RESULT, typename VALUE>
std::ostream &ConstantBase<RESULT, VALUE>::AsFortran(std::ostream &o) const {
  if (Rank() > 1) {
    o << "reshape(";
  }
  if (Rank() > 0) {
    o << '[' << GetType().AsFortran() << "::";
  }
  bool first{true};
  for (const auto &value : values_) {
    if (first) {
      first = false;
    } else {
      o << ',';
    }
    if constexpr (Result::category == TypeCategory::Integer) {
      o << value.SignedDecimal() << '_' << Result::kind;
    } else if constexpr (Result::category == TypeCategory::Real ||
        Result::category == TypeCategory::Complex) {
      value.AsFortran(o, Result::kind);
    } else if constexpr (Result::category == TypeCategory::Character) {
      o << Result::kind << '_' << parser::QuoteCharacterLiteral(value);
    } else if constexpr (Result::category == TypeCategory::Logical) {
      if (value.IsTrue()) {
        o << ".true.";
      } else {
        o << ".false.";
      }
      o << '_' << Result::kind;
    } else {
      StructureConstructor{AsConstant().derivedTypeSpec(), value}.AsFortran(o);
    }
  }
  if (Rank() > 0) {
    o << ']';
  }
  if (Rank() > 1) {
    o << ",shape=";
    char ch{'['};
    for (auto dim : shape_) {
      o << ch << dim;
      ch = ',';
    }
    o << "])";
  }
  return o;
}

template<typename RESULT, typename VALUE>
auto ConstantBase<RESULT, VALUE>::At(
    const std::vector<std::int64_t> &index) const -> Value {
  CHECK(index.size() == static_cast<std::size_t>(Rank()));
  std::int64_t stride{1}, offset{0};
  int dim{0};
  for (std::int64_t j : index) {
    std::int64_t bound{shape_[dim++]};
    CHECK(j >= 1 && j <= bound);
    offset += stride * (j - 1);
    stride *= bound;
  }
  return values_.at(offset);
}

template<typename RESULT, typename VALUE>
Constant<SubscriptInteger> ConstantBase<RESULT, VALUE>::SHAPE() const {
  using IntType = Scalar<SubscriptInteger>;
  std::vector<IntType> result;
  for (std::int64_t dim : shape_) {
    result.emplace_back(dim);
  }
  return {std::move(result), std::vector<std::int64_t>{Rank()}};
}

Constant<SomeDerived>::Constant(const StructureConstructor &x)
  : Base{x.values()}, derivedTypeSpec_{&x.derivedTypeSpec()} {}

Constant<SomeDerived>::Constant(StructureConstructor &&x)
  : Base{std::move(x.values())}, derivedTypeSpec_{&x.derivedTypeSpec()} {}

Constant<SomeDerived>::Constant(const semantics::DerivedTypeSpec &spec,
    std::vector<StructureConstructorValues> &&x, std::vector<std::int64_t> &&s)
  : Base{std::move(x), std::move(s)}, derivedTypeSpec_{&spec} {}

static std::vector<StructureConstructorValues> GetValues(
    std::vector<StructureConstructor> &&x) {
  std::vector<StructureConstructorValues> result;
  for (auto &&structure : std::move(x)) {
    result.emplace_back(std::move(structure.values()));
  }
  return result;
}

Constant<SomeDerived>::Constant(const semantics::DerivedTypeSpec &spec,
    std::vector<StructureConstructor> &&x, std::vector<std::int64_t> &&s)
  : Base{GetValues(std::move(x)), std::move(s)}, derivedTypeSpec_{&spec} {}

FOR_EACH_INTRINSIC_KIND(template class ConstantBase)
template class ConstantBase<SomeDerived, StructureConstructorValues>;
FOR_EACH_INTRINSIC_KIND(template class Constant)
}
