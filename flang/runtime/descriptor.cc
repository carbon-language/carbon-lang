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
#include <new>

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

Descriptor::Descriptor(TypeCode t, std::size_t elementBytes, void *p, int rank,
    const SubscriptValue *extent) {
  CFI_establish(
      &raw_, p, CFI_attribute_other, t.raw(), elementBytes, rank, extent);
}

Descriptor::Descriptor(TypeCode::Form f, int kind, void *p, int rank,
    const SubscriptValue *extent) {
  std::size_t elementBytes = kind;
  if (f == TypeCode::Form::Complex) {
    elementBytes *= 2;
  }
  ISO::CFI_establish(&raw_, p, CFI_attribute_other, TypeCode(f, kind).raw(),
      elementBytes, rank, extent);
}

Descriptor::Descriptor(const DerivedTypeSpecialization &dts, void *p, int rank,
    const SubscriptValue *extent) {
  ISO::CFI_establish(
      &raw_, p, ADDENDUM, CFI_type_struct, dts.SizeInBytes(), rank, extent);
  Addendum()->set_derivedTypeSpecialization(dts);
}

Descriptor *Descriptor::Create(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent) {
  return new (new char[SizeInBytes(rank)])
      Descriptor{t, elementBytes, p, rank, extent};
}

Descriptor *Descriptor::Create(TypeCode::Form f, int kind, void *p, int rank,
    const SubscriptValue *extent) {
  return new (new char[SizeInBytes(rank)]) Descriptor{f, kind, p, rank, extent};
}

Descriptor *Descriptor::Create(const DerivedTypeSpecialization &dts, void *p,
    int rank, const SubscriptValue *extent) {
  const DerivedType &derivedType{dts.derivedType()};
  return new (new char[SizeInBytes(rank, derivedType.IsNontrivial(),
      derivedType.lenParameters())]) Descriptor{dts, p, rank, extent};
}

void Descriptor::Destroy() { delete[] reinterpret_cast<char *>(this); }

void Descriptor::SetDerivedTypeSpecialization(
    const DerivedTypeSpecialization &dts) {
  raw_.attribute |= ADDENDUM;
  Addendum()->set_derivedTypeSpecialization(dts);
}

void Descriptor::SetLenParameterValue(int which, TypeParameterValue x) {
  raw_.attribute |= ADDENDUM;
  Addendum()->SetLenParameterValue(which, x);
}

std::size_t Descriptor::SizeInBytes() const {
  const DescriptorAddendum *addendum{Addendum()};
  return sizeof *this - sizeof(Dimension) + raw_.rank * sizeof(Dimension) +
      (addendum ? addendum->SizeInBytes() : 0);
}

TypeParameterValue TypeParameter::KindParameterValue(
    const DerivedTypeSpecialization &specialization) const {
  return specialization.KindParameterValue(which_);
}

TypeParameterValue TypeParameter::Value(const Descriptor &descriptor) const {
  const DescriptorAddendum &addendum{*descriptor.Addendum()};
  if (isLenTypeParameter_) {
    return addendum.LenParameterValue(which_);
  } else {
    return KindParameterValue(*addendum.derivedTypeSpecialization());
  }
}

bool DerivedType::IsNontrivialAnalysis() const {
  if (kindParameters_ > 0 || lenParameters_ > 0 || typeBoundProcedures_ > 0 ||
      definedAssignments_ > 0 || finalSubroutine_.host != 0) {
    return true;
  }
  for (int j{0}; j < components_; ++j) {
    if (component_[j].IsDescriptor()) {
      return true;
    }
    if (const Descriptor * staticDescriptor{component_[j].staticDescriptor()}) {
      if (const DescriptorAddendum * addendum{staticDescriptor->Addendum()}) {
        if (const DerivedTypeSpecialization *
            dts{addendum->derivedTypeSpecialization()}) {
          if (dts->derivedType().IsNontrivial()) {
            return true;
          }
        }
      }
    }
  }
  return false;
}
}  // namespace Fortran::runtime
