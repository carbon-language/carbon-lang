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

// TODO: Not complete; exists to check compilability of descriptor.h

#include "descriptor.h"
#include <cstdlib>

namespace Fortran::runtime {

TypeCode::TypeCode(TypeCode::Form f, int kind) {
  switch (f) {
  case Form::Integer:
    switch (kind) {
    case 1: raw_ = CFI_type_int8_t; break;
    case 2: raw_ = CFI_type_int16_t; break;
    case 4: raw_ = CFI_type_int32_t; break;
    case 8: raw_ = CFI_type_int64_t; break;
    case 16: raw_ = CFI_type_int128_t; break;
    }
    break;
  case Form::Real:
    switch (kind) {
    case 4: raw_ = CFI_type_float; break;
    case 8: raw_ = CFI_type_double; break;
    case 10:
    case 16: raw_ = CFI_type_long_double; break;
    }
    break;
  case Form::Complex:
    switch (kind) {
    case 4: raw_ = CFI_type_float_Complex; break;
    case 8: raw_ = CFI_type_double_Complex; break;
    case 10:
    case 16: raw_ = CFI_type_long_double_Complex; break;
    }
    break;
  case Form::Character:
    if (kind == 1) {
      raw_ = CFI_type_cptr;
    }
    break;
  case Form::Logical:
    switch (kind) {
    case 1: raw_ = CFI_type_Bool; break;
    case 2: raw_ = CFI_type_int16_t; break;
    case 4: raw_ = CFI_type_int32_t; break;
    case 8: raw_ = CFI_type_int64_t; break;
    }
    break;
  case Form::Derived: raw_ = CFI_type_struct; break;
  }
}

std::size_t DescriptorAddendum::SizeInBytes() const {
  return SizeInBytes(derivedTypeSpecialization_->derivedType().lenParameters());
}

void DescriptorView::SetDerivedTypeSpecialization(
    const DerivedTypeSpecialization &dts) {
  raw_.attribute |= ADDENDUM;
  Addendum()->set_derivedTypeSpecialization(dts);
}

void DescriptorView::SetLenParameterValue(int which, TypeParameterValue x) {
  raw_.attribute |= ADDENDUM;
  Addendum()->SetLenParameterValue(which, x);
}

std::size_t DescriptorView::SizeInBytes() const {
  const DescriptorAddendum *addendum{Addendum()};
  return sizeof *this - sizeof(Dimension) + raw_.rank * sizeof(Dimension) +
      (addendum ? addendum->SizeInBytes() : 0);
}

int DescriptorView::Establish(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent) {
  return CFI_establish(
      &raw_, p, CFI_attribute_other, t.raw(), elementBytes, rank, extent);
}

int DescriptorView::Establish(TypeCode::Form f, int kind, void *p, int rank,
    const SubscriptValue *extent) {
  std::size_t elementBytes = kind;
  if (f == TypeCode::Form::Complex) {
    elementBytes *= 2;
  }
  return ISO::CFI_establish(&raw_, p, CFI_attribute_other,
      TypeCode(f, kind).raw(), elementBytes, rank, extent);
}

int DescriptorView::Establish(const DerivedTypeSpecialization &dts, void *p,
    int rank, const SubscriptValue *extent) {
  int result{ISO::CFI_establish(
      &raw_, p, ADDENDUM, CFI_type_struct, dts.SizeInBytes(), rank, extent)};
  if (result == CFI_SUCCESS) {
    Addendum()->set_derivedTypeSpecialization(dts);
  }
  return result;
}

TypeParameterValue TypeParameter::KindParameterValue(
    const DerivedTypeSpecialization &specialization) const {
  return specialization.KindParameterValue(which_);
}

TypeParameterValue TypeParameter::Value(
    const DescriptorView &descriptor) const {
  const DescriptorAddendum &addendum{*descriptor.Addendum()};
  if (isLenTypeParameter_) {
    return addendum.LenParameterValue(which_);
  } else {
    return KindParameterValue(*addendum.derivedTypeSpecialization());
  }
}

bool DerivedType::IsNonTrivial() const {
  if (kindParameters_ > 0 || lenParameters_ > 0 || typeBoundProcedures_ > 0 ||
      definedAssignments_ > 0 || finalSubroutine_.host != 0) {
    return true;
  }
  for (int j{0}; j < components_; ++j) {
    if (component_[j].IsDescriptor()) {
      return true;
    }
    if (const DescriptorView *
        staticDescriptor{component_[j].staticDescriptor()}) {
      if (const DescriptorAddendum * addendum{staticDescriptor->Addendum()}) {
        if (const DerivedTypeSpecialization *
            dts{addendum->derivedTypeSpecialization()}) {
          if (dts->derivedType().IsNonTrivial()) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

Object::~Object() {
  if (p_ != nullptr) {
    // TODO final procedure calls and component destruction
    delete reinterpret_cast<char *>(p_);
    p_ = nullptr;
  }
}

bool Object::Create(
    TypeCode::Form f, int kind, int rank, const SubscriptValue *extent) {
  if (f == TypeCode::Form::Character || f == TypeCode::Form::Derived) {
    // TODO support these...
    return false;
  }
  std::size_t descriptorBytes{DescriptorView::SizeInBytes(rank)};
  std::size_t elementBytes = kind;
  if (f == TypeCode::Form::Complex) {
    elementBytes *= 2;
  }
  std::size_t elements{1};
  for (int j{0}; j < rank; ++j) {
    if (extent[j] < 0) {
      return false;
    }
    elements *= static_cast<std::size_t>(extent[j]);
  }
  std::size_t totalBytes{descriptorBytes + elements * elementBytes};
  char *p{reinterpret_cast<char *>(std::malloc(totalBytes))};
  if (p == nullptr) {
    return false;
  }
  p_ = reinterpret_cast<DescriptorView *>(p);
  p_->Establish(f, kind, p + descriptorBytes, rank, extent);
  return true;
}
}  // namespace Fortran::runtime
