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
#include <algorithm>

namespace Fortran::evaluate {

template<typename RESULT, typename VALUE>
ConstantBase<RESULT, VALUE>::~ConstantBase() {}

static void ShapeAsFortran(
    std::ostream &o, const std::vector<std::int64_t> &shape) {
  if (shape.size() > 1) {
    o << ",shape=";
    char ch{'['};
    for (auto dim : shape) {
      o << ch << dim;
      ch = ',';
    }
    o << "])";
  }
}

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
  ShapeAsFortran(o, shape_);
  return o;
}

static std::int64_t SubscriptsToOffset(const std::vector<std::int64_t> &index,
    const std::vector<std::int64_t> &shape) {
  CHECK(index.size() == shape.size());
  std::int64_t stride{1}, offset{0};
  int dim{0};
  for (std::int64_t j : index) {
    std::int64_t bound{shape[dim++]};
    CHECK(j >= 1 && j <= bound);
    offset += stride * (j - 1);
    stride *= bound;
  }
  return offset;
}

template<typename RESULT, typename VALUE>
auto ConstantBase<RESULT, VALUE>::At(
    const std::vector<std::int64_t> &index) const -> ScalarValue {
  return values_.at(SubscriptsToOffset(index, shape_));
}

static Constant<SubscriptInteger> ShapeAsConstant(
    const std::vector<std::int64_t> &shape) {
  using IntType = Scalar<SubscriptInteger>;
  std::vector<IntType> result;
  for (std::int64_t dim : shape) {
    result.emplace_back(dim);
  }
  return {std::move(result),
      std::vector<std::int64_t>{static_cast<std::int64_t>(shape.size())}};
}

template<typename RESULT, typename VALUE>
Constant<SubscriptInteger> ConstantBase<RESULT, VALUE>::SHAPE() const {
  return ShapeAsConstant(shape_);
}

// Constant<Type<TypeCategory::Character, KIND>  specializations

template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(const ScalarValue &str)
  : values_{str}, length_{static_cast<std::int64_t>(values_.size())} {}

template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(ScalarValue &&str)
  : values_{std::move(str)}, length_{
                                 static_cast<std::int64_t>(values_.size())} {}

template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(std::int64_t len,
    std::vector<ScalarValue> &&strings, std::vector<std::int64_t> &&dims)
  : length_{len} {
  values_.assign(strings.size() * length_,
      static_cast<typename ScalarValue::value_type>(' '));
  std::int64_t at{0};
  for (const auto &str : strings) {
    values_.replace(
        at, std::min(length_, static_cast<std::int64_t>(str.size())), str);
    at += length_;
  }
  CHECK(at == static_cast<std::int64_t>(values_.size()));
}

template<int KIND> Constant<Type<TypeCategory::Character, KIND>>::~Constant() {}

template<int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::At(
    const std::vector<std::int64_t> &index) const -> ScalarValue {
  auto offset{SubscriptsToOffset(index, shape_)};
  return values_.substr(offset, length_);
}

template<int KIND>
Constant<SubscriptInteger>
Constant<Type<TypeCategory::Character, KIND>>::SHAPE() const {
  return ShapeAsConstant(shape_);
}

template<int KIND>
std::ostream &Constant<Type<TypeCategory::Character, KIND>>::AsFortran(
    std::ostream &o) const {
  if (Rank() > 1) {
    o << "reshape(";
  }
  if (Rank() > 0) {
    o << '[' << GetType().AsFortran() << "::";
  }
  bool first{true};
  auto total{static_cast<std::int64_t>(size())};
  for (std::int64_t at{0}; at < total; at += length_) {
    ScalarValue value{values_.substr(at, length_)};
    if (first) {
      first = false;
    } else {
      o << ',';
    }
    o << Result::kind << '_' << parser::QuoteCharacterLiteral(value);
  }
  if (Rank() > 0) {
    o << ']';
  }
  ShapeAsFortran(o, shape_);
  return o;
}

// Constant<SomeDerived> specialization

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

FOR_EACH_LENGTHLESS_INTRINSIC_KIND(template class ConstantBase)
template class ConstantBase<SomeDerived, StructureConstructorValues>;
FOR_EACH_INTRINSIC_KIND(template class Constant)
}
