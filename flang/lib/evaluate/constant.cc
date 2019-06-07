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
#include "shape.h"
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
  int rank{GetRank(shape)};
  CHECK(GetRank(indices) == rank);
  for (int j{0}; j < rank; ++j) {
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

template<typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::ConstantBase(
    std::vector<Element> &&x, ConstantSubscripts &&dims, Result res)
  : result_{res}, values_(std::move(x)), shape_(std::move(dims)) {
  CHECK(size() == TotalElementCount(shape_));
}

template<typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::~ConstantBase() {}

template<typename RESULT, typename ELEMENT>
bool ConstantBase<RESULT, ELEMENT>::operator==(const ConstantBase &that) const {
  return shape_ == that.shape_ && values_ == that.values_;
}

static ConstantSubscript SubscriptsToOffset(
    const ConstantSubscripts &index, const ConstantSubscripts &shape) {
  CHECK(GetRank(index) == GetRank(shape));
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

template<typename RESULT, typename ELEMENT>
Constant<SubscriptInteger> ConstantBase<RESULT, ELEMENT>::SHAPE() const {
  return AsConstantShape(shape_);
}

template<typename RESULT, typename ELEMENT>
auto ConstantBase<RESULT, ELEMENT>::Reshape(
    const ConstantSubscripts &dims) const -> std::vector<Element> {
  std::size_t n{TotalElementCount(dims)};
  CHECK(!empty() || n == 0);
  std::vector<Element> elements;
  auto iter{values().cbegin()};
  while (n-- > 0) {
    elements.push_back(*iter);
    if (++iter == values().cend()) {
      iter = values().cbegin();
    }
  }
  return elements;
}

template<typename T>
auto Constant<T>::At(const ConstantSubscripts &index) const -> Element {
  return Base::values_.at(SubscriptsToOffset(index, Base::shape_));
}

template<typename T>
auto Constant<T>::Reshape(ConstantSubscripts &&dims) const -> Constant {
  return {Base::Reshape(dims), std::move(dims)};
}

// Constant<Type<TypeCategory::Character, KIND> specializations
template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(
    const Scalar<Result> &str)
  : values_{str}, length_{static_cast<ConstantSubscript>(values_.size())} {}

template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(Scalar<Result> &&str)
  : values_{std::move(str)}, length_{static_cast<ConstantSubscript>(
                                 values_.size())} {}

template<int KIND>
Constant<Type<TypeCategory::Character, KIND>>::Constant(ConstantSubscript len,
    std::vector<Scalar<Result>> &&strings, ConstantSubscripts &&dims)
  : length_{len}, shape_{std::move(dims)} {
  CHECK(strings.size() == TotalElementCount(shape_));
  values_.assign(strings.size() * length_,
      static_cast<typename Scalar<Result>::value_type>(' '));
  ConstantSubscript at{0};
  for (const auto &str : strings) {
    auto strLen{static_cast<ConstantSubscript>(str.size())};
    if (strLen > length_) {
      values_.replace(at, length_, str.substr(0, length_));
    } else {
      values_.replace(at, strLen, str);
    }
    at += length_;
  }
  CHECK(at == static_cast<ConstantSubscript>(values_.size()));
}

template<int KIND> Constant<Type<TypeCategory::Character, KIND>>::~Constant() {}

template<int KIND>
bool Constant<Type<TypeCategory::Character, KIND>>::empty() const {
  return size() == 0;
}

template<int KIND>
std::size_t Constant<Type<TypeCategory::Character, KIND>>::size() const {
  if (length_ == 0) {
    return TotalElementCount(shape_);
  } else {
    return static_cast<ConstantSubscript>(values_.size()) / length_;
  }
}

template<int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::At(
    const ConstantSubscripts &index) const -> Scalar<Result> {
  auto offset{SubscriptsToOffset(index, shape_)};
  return values_.substr(offset * length_, length_);
}

template<int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::Reshape(
    ConstantSubscripts &&dims) const -> Constant<Result> {
  std::size_t n{TotalElementCount(dims)};
  CHECK(!empty() || n == 0);
  std::vector<Element> elements;
  ConstantSubscript at{0},
      limit{static_cast<ConstantSubscript>(values_.size())};
  while (n-- > 0) {
    elements.push_back(values_.substr(at, length_));
    at += length_;
    if (at == limit) {  // subtle: at > limit somehow? substr() will catch it
      at = 0;
    }
  }
  return {length_, std::move(elements), std::move(dims)};
}

template<int KIND>
Constant<SubscriptInteger>
Constant<Type<TypeCategory::Character, KIND>>::SHAPE() const {
  return AsConstantShape(shape_);
}

// Constant<SomeDerived> specialization
Constant<SomeDerived>::Constant(const StructureConstructor &x)
  : Base{x.values(), Result{x.derivedTypeSpec()}} {}

Constant<SomeDerived>::Constant(StructureConstructor &&x)
  : Base{std::move(x.values()), Result{x.derivedTypeSpec()}} {}

Constant<SomeDerived>::Constant(const semantics::DerivedTypeSpec &spec,
    std::vector<StructureConstructorValues> &&x, ConstantSubscripts &&s)
  : Base{std::move(x), std::move(s), Result{spec}} {}

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
  : Base{AcquireValues(std::move(x)), std::move(s), Result{spec}} {}

std::optional<StructureConstructor>
Constant<SomeDerived>::GetScalarValue() const {
  if (shape_.empty()) {
    return StructureConstructor{result().derivedTypeSpec(), values_.at(0)};
  } else {
    return std::nullopt;
  }
}

StructureConstructor Constant<SomeDerived>::At(
    const ConstantSubscripts &index) const {
  return {result().derivedTypeSpec(),
      values_.at(SubscriptsToOffset(index, shape_))};
}

auto Constant<SomeDerived>::Reshape(ConstantSubscripts &&dims) const
    -> Constant {
  return {result().derivedTypeSpec(), Base::Reshape(dims), std::move(dims)};
}

INSTANTIATE_CONSTANT_TEMPLATES
}
