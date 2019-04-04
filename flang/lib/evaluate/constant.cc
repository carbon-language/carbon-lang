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
#include <string>

namespace Fortran::evaluate {

template<typename RESULT, typename VALUE>
ConstantBase<RESULT, VALUE>::~ConstantBase() {}

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

template<typename RESULT, typename VALUE>
auto ConstantBase<RESULT, VALUE>::At(std::vector<std::int64_t> &&index) const
    -> ScalarValue {
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

// Constant<Type<TypeCategory::Character, KIND> specializations
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
  : length_{len}, shape_{std::move(dims)} {
  values_.assign(strings.size() * length_,
      static_cast<typename ScalarValue::value_type>(' '));
  std::int64_t at{0};
  for (const auto &str : strings) {
    auto strLen{static_cast<std::int64_t>(str.size())};
    if (strLen > length_) {
      values_.replace(at, length_, str.substr(0, length_));
    } else {
      values_.replace(at, strLen, str);
    }
    at += length_;
  }
  CHECK(at == static_cast<std::int64_t>(values_.size()));
}

template<int KIND> Constant<Type<TypeCategory::Character, KIND>>::~Constant() {}

static std::int64_t ShapeElements(const std::vector<std::int64_t> &shape) {
  std::int64_t elements{1};
  for (auto dim : shape) {
    elements *= dim;
  }
  return elements;
}

template<int KIND>
bool Constant<Type<TypeCategory::Character, KIND>>::empty() const {
  return size() == 0;
}

template<int KIND>
std::size_t Constant<Type<TypeCategory::Character, KIND>>::size() const {
  if (length_ == 0) {
    return ShapeElements(shape_);
  } else {
    return static_cast<std::int64_t>(values_.size()) / length_;
  }
}

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

INSTANTIATE_CONSTANT_TEMPLATES
}
