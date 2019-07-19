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

ConstantBounds::ConstantBounds(const ConstantSubscripts &shape)
  : shape_(shape), lbounds_(shape_.size(), 1) {}

ConstantBounds::ConstantBounds(ConstantSubscripts &&shape)
  : shape_(std::move(shape)), lbounds_(shape_.size(), 1) {}

ConstantBounds::~ConstantBounds() = default;

void ConstantBounds::set_lbounds(ConstantSubscripts &&lb) {
  CHECK(lb.size() == shape_.size());
  lbounds_ = std::move(lb);
}

Constant<SubscriptInteger> ConstantBounds::SHAPE() const {
  return AsConstantShape(shape_);
}

ConstantSubscript ConstantBounds::SubscriptsToOffset(
    const ConstantSubscripts &index) const {
  CHECK(GetRank(index) == GetRank(shape_));
  ConstantSubscript stride{1}, offset{0};
  int dim{0};
  for (auto j : index) {
    auto lb{lbounds_[dim]};
    auto extent{shape_[dim++]};
    CHECK(j >= lb && j < lb + extent);
    offset += stride * (j - lb);
    stride *= extent;
  }
  return offset;
}

bool ConstantBounds::IncrementSubscripts(ConstantSubscripts &indices) const {
  int rank{GetRank(shape_)};
  CHECK(GetRank(indices) == rank);
  for (int j{0}; j < rank; ++j) {
    auto lb{lbounds_[j]};
    CHECK(indices[j] >= lb);
    if (++indices[j] < lb + shape_[j]) {
      return true;
    } else {
      CHECK(indices[j] == lb + shape_[j]);
      indices[j] = lb;
    }
  }
  return false;  // all done
}

template<typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::ConstantBase(
    std::vector<Element> &&x, ConstantSubscripts &&sh, Result res)
  : ConstantBounds(std::move(sh)), result_{res}, values_(std::move(x)) {
  CHECK(size() == TotalElementCount(shape()));
}

template<typename RESULT, typename ELEMENT>
ConstantBase<RESULT, ELEMENT>::~ConstantBase() {}

template<typename RESULT, typename ELEMENT>
bool ConstantBase<RESULT, ELEMENT>::operator==(const ConstantBase &that) const {
  return shape() == that.shape() && values_ == that.values_;
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
  return Base::values_.at(Base::SubscriptsToOffset(index));
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
    std::vector<Scalar<Result>> &&strings, ConstantSubscripts &&sh)
  : ConstantBounds(std::move(sh)), length_{len} {
  CHECK(strings.size() == TotalElementCount(shape()));
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
    return TotalElementCount(shape());
  } else {
    return static_cast<ConstantSubscript>(values_.size()) / length_;
  }
}

template<int KIND>
auto Constant<Type<TypeCategory::Character, KIND>>::At(
    const ConstantSubscripts &index) const -> Scalar<Result> {
  auto offset{SubscriptsToOffset(index)};
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
    std::vector<StructureConstructor> &&x, ConstantSubscripts &&shape)
  : Base{AcquireValues(std::move(x)), std::move(shape), Result{spec}} {}

std::optional<StructureConstructor>
Constant<SomeDerived>::GetScalarValue() const {
  if (Rank() == 0) {
    return StructureConstructor{result().derivedTypeSpec(), values_.at(0)};
  } else {
    return std::nullopt;
  }
}

StructureConstructor Constant<SomeDerived>::At(
    const ConstantSubscripts &index) const {
  return {result().derivedTypeSpec(), values_.at(SubscriptsToOffset(index))};
}

auto Constant<SomeDerived>::Reshape(ConstantSubscripts &&dims) const
    -> Constant {
  return {result().derivedTypeSpec(), Base::Reshape(dims), std::move(dims)};
}

INSTANTIATE_CONSTANT_TEMPLATES
}
