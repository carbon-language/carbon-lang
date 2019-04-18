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

std::size_t TotalElementCount(const ConstantSubscripts &shape) {
  std::size_t size{1};
  for (auto dim : shape) {
    CHECK(dim >= 0);
    size *= dim;
  }
  return size;
}

bool IncrementSubscripts(
    ConstantSubscripts &indices, const ConstantSubscripts &shape) {
  auto rank{shape.size()};
  CHECK(indices.size() == rank);
  for (std::size_t j{0}; j < rank; ++j) {
    CHECK(indices[j] >= 1);
    if (++indices[j] <= shape[j]) {
      return true;
    } else {
      CHECK(indices[j] == shape[j] + 1);
      indices[j] = 1;
    }
  }
  return false;  // all done
}

template<typename RESULT, typename VALUE>
ConstantBase<RESULT, VALUE>::~ConstantBase() {}

static ConstantSubscript SubscriptsToOffset(
    const ConstantSubscripts &index, const ConstantSubscripts &shape) {
  CHECK(index.size() == shape.size());
  ConstantSubscript stride{1}, offset{0};
  int dim{0};
  for (auto j : index) {
    auto bound{shape[dim++]};
    CHECK(j >= 1 && j <= bound);
    offset += stride * (j - 1);
    stride *= bound;
  }
  return offset;
}

template<typename RESULT, typename VALUE>
auto ConstantBase<RESULT, VALUE>::At(const ConstantSubscripts &index) const
    -> ScalarValue {
  return values_.at(SubscriptsToOffset(index, shape_));
}

template<typename RESULT, typename VALUE>
auto ConstantBase<RESULT, VALUE>::At(ConstantSubscripts &&index) const
    -> ScalarValue {
  return values_.at(SubscriptsToOffset(index, shape_));
}

static Constant<SubscriptInteger> ShapeAsConstant(
    const ConstantSubscripts &shape) {
  using IntType = Scalar<SubscriptInteger>;
  std::vector<IntType> result;
  for (auto dim : shape) {
    result.emplace_back(dim);
  }
  return {std::move(result),
      ConstantSubscripts{static_cast<std::int64_t>(shape.size())}};
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
    std::vector<ScalarValue> &&strings, ConstantSubscripts &&dims)
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

static ConstantSubscript ShapeElements(const ConstantSubscripts &shape) {
  ConstantSubscript elements{1};
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
    const ConstantSubscripts &index) const -> ScalarValue {
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
    std::vector<StructureConstructorValues> &&x, ConstantSubscripts &&s)
  : Base{std::move(x), std::move(s)}, derivedTypeSpec_{&spec} {}

static std::vector<StructureConstructorValues> AcquireValues(
    std::vector<StructureConstructor> &&x) {
  std::vector<StructureConstructorValues> result;
  for (auto &&structure : std::move(x)) {
    result.emplace_back(std::move(structure.values()));
  }
  return result;
}

Constant<SomeDerived>::Constant(const semantics::DerivedTypeSpec &spec,
    std::vector<StructureConstructor> &&x, ConstantSubscripts &&s)
  : Base{AcquireValues(std::move(x)), std::move(s)}, derivedTypeSpec_{&spec} {}

INSTANTIATE_CONSTANT_TEMPLATES
}
